import torch
import os
from maddpg.actor_critic import Actor, Critic

class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # 设置 device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建网络并移动到 device
        self.actor_network = Actor(args, agent_id).to(self.device)
        self.critic_network = Critic(args).to(self.device)

        # 创建目标网络并移动到 device
        self.actor_target_network = Actor(args, agent_id).to(self.device)
        self.critic_target_network = Critic(args).to(self.device)

        # 将当前网络参数复制到目标网络
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # 创建优化器
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # 保存模型路径
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = os.path.join(self.model_path, f'agent_{agent_id}')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载已有模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl', map_location=self.device))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl', map_location=self.device))
            print(f'Agent {self.agent_id} successfully loaded actor_network: {self.model_path}/actor_params.pkl')
            print(f'Agent {self.agent_id} successfully loaded critic_network: {self.model_path}/critic_params.pkl')

    # 软更新目标网络
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # 训练更新网络
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32, device=self.device)

        r = transitions[f'r_{self.agent_id}']  # 只用自己的reward
        o, u, o_next = [], [], []

        for agent_id in range(self.args.n_agents):
            o.append(transitions[f'o_{agent_id}'])
            u.append(transitions[f'u_{agent_id}'])
            o_next.append(transitions[f'o_next_{agent_id}'])

        # 计算 target Q 值
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # 计算 critic loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # 计算 actor loss
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()

        # 更新 actor 网络
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 更新 critic 网络
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 软更新目标网络
        self._soft_update_target_network()

        # 保存模型
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, f'agent_{self.agent_id}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + num + '_critic_params.pkl')
