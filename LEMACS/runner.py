from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import re
from maddpg.maddpg import MADDPG
from tqdm import trange
import json  # 或者使用其他模块如pickle, csv等根据你的需求
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.eval_episodes = args.eval_episodes
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        # 打印设备信息
        print("=" * 50)
        flag = torch.cuda.is_available()
        device = torch.device("cuda:0" if flag else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if flag else "无GPU可用"
        scenario = "Search_LLM_RL_v6.0.1"
        # 打印环境信息
        print(f"{'项'.ljust(15)} | {'值'}")
        print("-" * 50)
        print(f"{'CUDA状态'.ljust(15)} | {'可用' if flag else '不可用'}")
        print(f"{'设备'.ljust(15)} | {device}")
        print(f"{'GPU型号'.ljust(15)} | {gpu_name}")
        print(f"{'Scenario'.ljust(15)} | {scenario}")
        print("=" * 50)
        current_path = os.path.dirname(os.path.realpath(__file__))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        reward_dir = os.path.join(current_path, "models", scenario, "Plt5.2", timestamp)
        os.makedirs(reward_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=reward_dir)
        total_episodes = self.args.time_steps // self.args.max_episode_len
        print(f"Total episodes: {total_episodes}")
        print("=" * 50)
        for episode in range(total_episodes):  # 修改为普通的for循环，去掉进度条
            s = self.env.reset()
            single_agent_rewards = [0] * self.args.n_agents
            all_episode_rewards = []
            all_episode_rewards_test = []
            my_dict = {i: [] for i in range(self.args.n_agents)}
            my_state = {i: [] for i in range(self.args.n_agents)}
            tag = [0] * self.args.n_agents
            for step in range(self.episode_limit):
                u = []
                actions = []
                episode_rewards = [0] * self.args.n_agents
                episode_rewards_test = [0] * self.args.n_agents
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)
                s_next, r, done, info = self.env.step(actions)
                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                          s_next[:self.args.n_agents])
                for agent_id in range(self.args.n_agents):
                    my_state[agent_id].append(s_next[agent_id][2:4])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)
                self.noise = max(0.05, self.noise - 5e-7)
                self.epsilon = max(0.05, self.epsilon - 5e-7)
                for agent_id in range(self.args.n_agents):
                    if r[agent_id] > 70 and tag[agent_id] == 0:
                        for a in range(self.args.n_agents):
                            now_step = step
                            single_agent_rewards[a] = now_step
                            if a == agent_id:
                                episode_rewards[a] = r[a]
                                episode_rewards_test[a] = 100
                            else:
                                episode_rewards[a] = r[a] + 100
                                episode_rewards_test[a] = 100
                        tag = [1] * self.args.n_agents
                    elif r[agent_id] < 90:
                        if (step + 1) == self.episode_limit and tag[agent_id] == 0:
                            single_agent_rewards[agent_id] = self.episode_limit
                        episode_rewards[agent_id] = r[agent_id]
                        episode_rewards_test[agent_id] = -1
                    else:
                        episode_rewards[agent_id] = 0
                        episode_rewards_test[agent_id] = 0
                    my_dict[agent_id].append(episode_rewards[agent_id])
                    writer.add_scalar(f'Step_single_agent_reward/{agent_id}_reward', r[agent_id],
                                      episode * self.episode_limit + step)
                all_episode_rewards.append(sum(episode_rewards))
                all_episode_rewards_test.append(sum(episode_rewards_test))
                if any(tag == 0 for tag in done):
                    print(f"\n[Episode {episode + 1}] 提前结束，step: {step}")
                    break                 # 🔥加break，才能真正跳出for循环
            # 写入Tensorboard统计
            print(f"episode:{episode}")
            total_episode_reward = sum(all_episode_rewards)
            total_episode_reward_test = sum(all_episode_rewards_test)
            writer.add_scalar('Episode_all_agent_reward', total_episode_reward, episode)
            writer.add_scalar('Episode_all_agent_reward_test', total_episode_reward_test, episode)
            collected_rewards = []
            for agent_id in range(self.args.n_agents):
                total_sum = sum(my_dict[agent_id])
                collected_rewards.append(single_agent_rewards[agent_id])
                writer.add_scalar(f'Episode_single_agent_{agent_id}/{agent_id}_step', single_agent_rewards[agent_id],
                                  episode)
                writer.add_scalar(f'Episode_single_agent_{agent_id}/{agent_id}_total_reward', total_sum, episode)
            min_reward = min(collected_rewards)
            writer.add_scalar('Episode_Min_Step', min_reward, episode)

    def evaluate(self, model_dir="new-model4/simple_spread"):
        print("Evaluating pretrained model...")
        # 用于存储每个 agent 的 MADDPG 实例（只用于加载和评估）
        maddpg_models = []
        # 获取设备信息，确保所有操作都在相同的设备上
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 遍历加载每个 agent 的 actor 网络
        for agent_id in range(self.args.n_agents):
            agent_model_dir = os.path.join(model_dir, f"agent_{agent_id}")
            if not os.path.exists(agent_model_dir):
                print(f"[Warning] Directory not found: {agent_model_dir}")
                maddpg_models.append(None)
                continue
            model_files = [f for f in os.listdir(agent_model_dir) if re.match(r'\d+_actor_params\.pkl', f)]
            if not model_files:
                print(f"[Warning] No actor models found in {agent_model_dir}")
                maddpg_models.append(None)
                continue
            latest_file = sorted(model_files, key=lambda x: int(x.split('_')[0]))[-1]
            actor_path = os.path.join(agent_model_dir, latest_file)
            # 创建 MADDPG 实例并加载模型
            maddpg = MADDPG(self.args, agent_id)
            maddpg.actor_network.load_state_dict(torch.load(actor_path, map_location=device))  # 将模型加载到相应设备
            maddpg.actor_network.to(device)  # 确保模型在正确的设备上
            maddpg.actor_network.eval()
            maddpg_models.append(maddpg)
            print(f"[Loaded] agent_{agent_id}: {actor_path}")
        # 开始评估
        returns = []
        for episode in range(self.eval_episodes):
            s = self.env.reset()
            episode_reward = 0
            for time_step in range(self.episode_limit):
                actions = []
                with torch.no_grad():
                    for agent_id in range(self.args.n_agents):
                        obs = torch.tensor(s[agent_id], dtype=torch.float32).unsqueeze(0)
                        obs = obs.to(device)  # 确保 obs 在正确的设备上
                        maddpg = maddpg_models[agent_id]
                        if maddpg is not None:
                            action = maddpg.actor_network(obs).squeeze(0).cpu().numpy()  # .cpu() 将张量从 GPU 转移到 CPU
                        else:
                            action = np.zeros(self.args.action_shape[agent_id])  # 若模型未加载，填充0动作
                        actions.append(action)
                # 如果有额外的玩家，添加相应的动作
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                episode_reward += sum(r[:self.args.n_agents])
                s = s_next
                if any(tag == 0 for tag in done):
                    # Temp = self.episode_limit - (time_step % self.episode_limit)
                    # print(f"Temp：{Temp}")
                    print(f"有agent完成任务，提前结束本轮！done = {done}")
                    break  # <--- 🔥加break，才能真正跳出for循环
