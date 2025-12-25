# -*- coding: utf-8 -*-
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import matplotlib.pyplot as plt
import time
import numpy as np
import re
import csv
import random
from openai import OpenAI
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.episode = -2
        self.probability_map = None
        self.max_visits = 3
        self.detection_prob = 0.95
        self.false_alarm_prob = 0.05
        self.numx = 0
        self.visited_map = None
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.timestamp = time.strftime("%Y%m%d%H%M%S")
        self.reward_dir = os.path.join(self.current_path, "Map(Search_LLM-RL(simple_spread_v6.0.1))", self.timestamp)
        os.makedirs(self.reward_dir, exist_ok=True)

    def make_world(self):
        world = World()
        square_centers = [
            [-2.5, 2],
            [-2.5, 2],
            [-2.5, 2],
        ]
        square_side_length = 10
        x_range = (-5, 5)
        y_range = (-5, 5)
        world.dim_c = 2
        num_agents = 3
        # num_agents = 6
        # num_agents = 9
        num_landmarks = 9
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.01
            agent.tag = 4
            agent.max = 0
            agent.ref = 0
            agent.comm_channel = []
            agent.his_comm_channel = {}
            agent.his_comm_channel_p_vel = {}
            agent.his_comm_channel_p_pos = {}
            agent.detection_range = 0.1
            agent.observation1 = 0
            agent.sigma = 5
            agent.probability_map = self.create_probability_map_with_squares(x_range, y_range, square_centers,square_side_length)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.occupied = False
            landmark.tag =4
            landmark.ref = 0
        self.numx = 0
        self.reset_world(world)
        return world

    def reset_world(self, world):
        square_centers_llm = [
            [-2.5, 2], [-2.5, 2], [-2.5, 2],
            [-1, 3.5], [-1, 3.5], [-1, 3.5],
            [-1, 3], [-1, 3], [-1, 3],
            [-1, 3], [-1, 3], [-1, 3],
        ]
        square_side_length_llm = [3, 2, 1, 0.5]

        x_range = (-5, 5)
        y_range = (-5, 5)
        self.episode += 1
        self.numx = 0
        square_centers = square_centers_llm[:3]
        square_side_length = square_side_length_llm[0]
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # 为每个地标设置随机的颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        init_agent_positions = [
            [4.7, -3.8], [4.5, -4.5], [4.6, -4.6],
        ]
        # 3个IOP   LLM知识注入
        with open('D:\Study\Code\V5\SearchLLM\output2.txt', 'r') as file:
            content = file.read()
        coordinates = re.findall(r'\[([-\d.]+),\s*([-\d.]+)\]', content)
        Planning_landmark_positions = [[float(x), float(y)] for x, y in coordinates]
        init_landmark_positions = [
            [0, 0], [0, 0], [0, 0],
            [-1, -4], [2, 1.5], [2.5, -1.5],
            [-0.95, 3.05], [-0.95, 3.05], [-0.95, 3.05],
            [-2.6, 0], [1.3, -2], [1.3, 2.8]
        ]
        real_target_positions = self.generate_real_target_positions()
        if len(real_target_positions) >= 3:
            init_landmark_positions[6:9] = real_target_positions[:3]
        init_landmark_positions[:3] = square_centers[:3]

        for i,agent in enumerate(world.agents):
            agent.state.p_pos = np.array(init_agent_positions[i])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.movable = True
            agent.tag = 4
            agent.c_energy = 0
            agent.max = 0
            agent.ref = 0
            agent.comm_channel = []
            agent.his_comm_channel = {}
            agent.his_comm_channel_p_vel = {}
            agent.his_comm_channel_p_pos = {}
            agent.probability_at_point1 = 0
            agent.observation1 = 0
            agent.probability_map = self.create_probability_map_with_squares(x_range, y_range, square_centers, square_side_length)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array(init_landmark_positions[i])
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.occupied = False
            landmark.tag = 4
            landmark.ref = 0
    def log_data(self, prompt, response_text, duration=None):
        with open("prediction_log4-05-13-(gpt-4o-mini)-15 ", "a", encoding="utf-8") as f:
            f.write(f"Prompt:\n{prompt}\n")
            f.write(f"Response:\n{response_text}\n")
            if duration is not None:
                f.write(f"Duration: {duration:.4f} seconds\n")
            f.write("=" * 50 + "\n")
    def predict_auv_trajectory(self, history_messages):
        client = OpenAI(
            api_key="OPENAI_API_KEY",
            base_url="OPENAI_BASE_URL"
        )
        history_str = "\n".join(history_messages)
        prompt = f"""You are an expert in the field of multi-agent cooperative search. This cluster consists of three agents: agent0, agent1, and agent2. When an agent loses contact due to exceeding the communication range, please use its **last known position**, **velocity direction**, and **historical communication data** to predict its next possible movement locations. The goal is to reduce the decline in cooperative search quality caused by communication instability between agents.
            The search environment is a 2D plane composed of 40×40 discrete grid cells. Coordinates are represented as (x, y), where x is the row number and y is the column number, and both range from 0 to 39.
            ### Agent Scanning Mechanism:
            Each agent's physical location is defined in continuous 2D space (e.g., [-5, 5] meters per axis). The scanning field-of-view (FOV) is determined based on **Euclidean distance** in this continuous space:
            > Each agent can **scan any grid points within a radius of 0.3 units** around its current physical position.
            This means an agent’s scan may cover more than just the four adjacent grid cells — potentially including diagonal or irregular nearby positions.
            ### AUV Motion Dynamics:
            Each AUV follows a damped motion model. At every time step, velocity decreases slightly due to physical drag:
                velocity = velocity × (1 - damping)
            The damping value is typically 0.25, so velocity decays gradually unless re-accelerated. The actual movement per step is:
                position = position + velocity × step_scale  
                where step_scale = 0.2 × dt
            This results in very small movements each step — usually requiring several steps for the agent to visibly change grid coordinates. Even with significant velocity, it may take multiple steps to move into a new grid cell.
            ### Velocity-to-Position Transition Example:
            To help you understand how **velocity evolution influences movement**, consider the following real-world position and velocity transitions of one agent:
            Step 1:  position: (38, 5), velocity: [-1.0000, 1.0000] 
            Step 2:  position: (38, 5), velocity: [-1.7500, 1.7500] 
            Step 3:  position: (38, 5), velocity: [-2.3125, 2.3125]  
            Step 4:  position: (37, 5), velocity: [-2.7344, 2.7344]
            Step 5:  position: (37, 5), velocity: [-3.0508, 3.0508]
            Step 6:  position: (37, 6), velocity: [-3.2881, 3.2881] 
            Step 7:  position: (37, 6), velocity: [-3.4661, 3.4661] 
            Step 8:  position: (36, 6), velocity: [-3.5995, 3.5995]
            Step 9:  position: (36, 6), velocity: [-3.6997, 3.6997]
            Step 10: position: (36, 7), velocity: [-3.7747, 3.7747]  
            From this pattern, we observe:
            - Velocity increases first, then gradually decays due to damping.
            - Position changes are not immediate — an agent may remain on the same grid for multiple steps despite changing velocity.
            - You must take both **current velocity and its damping trend** into account when predicting future positions.
            ### Communication History Influence:
            If the disconnected agent had a consistent movement pattern related to staying within communication range of other agents, this trend should be preserved. Use historical communication logs to infer typical movement intentions (e.g., moving east to track agent2).
            ### Your Task:
            For each **disconnected agent**, use all three types of input (position, velocity, communication history) to:
            1. Predict its next velocity direction.
            2. Predict the next likely grid position.
            3. Predict exactly four grid positions the agent may scan or move toward, along with a very small probability value indicating the likelihood.
            ### Input Information:
            You are given the following data for the disconnected agents: {history_str}
            - **Position information**: The last known position of the agent on the 2D grid (x_pos, y_pos), where x_pos and y_pos are integers in the range [0, 39]. All predicted x_pos and y_pos coordinates must strictly satisfy:  
              0 ≤ x_pos < 40 and 0 ≤ y_pos < 40  
              Never generate values outside of this range.
            - **Velocity direction**: A 2D velocity vector (vx, vy). The motion model of the AUV (autonomous underwater vehicle) follows the behavior style of `multi-agent.scenarios.simple_spread`, i.e., continuous and with gradual changes. Additionally, velocity may decay at each step due to damping.
            - **Historical communication data**: Most recent messages exchanged with other agents, indicating past coordination patterns and movement logic.
            ### Output Format:
            For each **disconnected** agent, output in **exactly** the following structure:
            agentX:  
            velocity: (vx, vy)  
            position: (x, y)  
            (x1, y1, p1)  
            (x2, y2, p2)  
            (x3, y3, p3)  
            (x4, y4, p4)  
            **Output Rules**:
            - agentX is the name (e.g., agent0, agent1, etc.).
            - vx, vy are floats for the predicted velocity direction.
            - (x, y) is the predicted next grid position (integers in [0, 39]).
            - Each of the four lines below it is a tuple: (x, y, p), where:
              - x and y are integers within [0, 39],
              - p is a float value between 0.00001 and 0.0001.
            - Do **not include** any explanation or commentary. Output results only.
        """
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model="llm-model",
                messages=[
                    {"role": "system", "content": "You are a professional underwater multi-agent tracking assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            end_time = time.time()
            duration = end_time - start_time
            print(f"预测耗时: {duration:.4f} 秒")
            response_text = completion.choices[0].message.content.strip()
            print("预测返回:", response_text.encode('utf-8').decode('utf-8'))
            self.log_data(prompt, response_text, duration)
        except Exception as e:
            print(f"调用LLM预测出错：{e}")
            return {}
        # 初始化结果字典
        predicted_info = {}
        velocity_info = {}
        position_info = {}
        agent_blocks = re.split(r'\n(?=agent[_\d]+:)', response_text)
        for block in agent_blocks:
            lines = block.strip().split('\n')
            if not lines or not lines[0].startswith('agent'):
                continue
            agent_name = lines[0].strip(':')
            # 匹配 velocity 和 position
            velocity_match = re.search(r'velocity:\s*\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', block)
            position_match = re.search(r'position:\s*\((\d+),\s*(\d+)\)', block)
            pred_matches = re.findall(r'\((\d+),\s*(\d+),\s*([\deE\+\.-]+)\)', block)
            if velocity_match:
                velocity_info[agent_name] = (float(velocity_match.group(1)), float(velocity_match.group(2)))
            if position_match:
                position_info[agent_name] = (int(position_match.group(1)), int(position_match.group(2)))
            if pred_matches:
                predicted_info[agent_name] = [(int(x), int(y), float(p)) for x, y, p in pred_matches]
        # 打印结构化信息
        print(f"LLM[预测结果] predicted_info: {predicted_info}")
        print(f"LLM[速度预测] velocity_info: {velocity_info}")
        print(f"LLM[位置预测] position_info: {position_info}")
        # 返回结构化信息
        return {
            "predicted_info": predicted_info,
            "velocity": velocity_info,
            "position": position_info
        }

    def normalize_agent_name(self, name):
        name = name.strip().lower().replace(":", "")
        return re.sub(r'agent(\d+)', r'agent_\1', name)

    def fuse_probability_maps_by_group(self, world, groups):
        for group in groups:
            if len(group) == 1:
                single_agent = group[0]
                print(f"当前分组只有一个AUV，调用LLM进行位置信息预测")
                history_messages = []
                for other_agent_name, messages in single_agent.his_comm_channel.items():
                    for message in messages[-4:]:
                        history_messages.append(f"{other_agent_name}: {message}")
                for other_agent_name in single_agent.his_comm_channel_p_vel:
                    vel_history = single_agent.his_comm_channel_p_vel[other_agent_name]
                    if vel_history:
                        last_vel = vel_history[-1]
                        history_messages.append(f"{other_agent_name} not in range. Last velocity: {last_vel}")
                for other_agent_name in single_agent.his_comm_channel_p_pos:
                    pos_history = single_agent.his_comm_channel_p_pos[other_agent_name]
                    if pos_history:
                        last_pos = pos_history[-1]
                        history_messages.append(f"{other_agent_name} not in range. Last position: {last_pos}")
                if history_messages:
                    result = self.predict_auv_trajectory(history_messages)
                    predicted_info = result.get("predicted_info", {})
                    velocity_info = result.get("velocity", {})
                    position_info = result.get("position", {})
                    for raw_name in predicted_info.keys():
                        other_name = self.normalize_agent_name(raw_name)
                        if other_name in single_agent.his_comm_channel:
                            single_agent.his_comm_channel[other_name].extend(predicted_info[raw_name])
                            # single_agent.his_comm_channel[other_name].append(predicted_info[raw_name])
                        if other_name in single_agent.his_comm_channel_p_vel:
                            single_agent.his_comm_channel_p_vel[other_name].append(velocity_info.get(raw_name))
                        if other_name in single_agent.his_comm_channel_p_pos:
                            single_agent.his_comm_channel_p_pos[other_name].append(position_info.get(raw_name))
                    for agent in group:
                        for target_name, predictions in predicted_info.items():
                            for x, y, prob in predictions:
                                x = max(0, min(x, 39))
                                y = max(0, min(y, 39))
                                # 修改位置：避免重复添加
                                if (x, y, round(prob, 5)) not in agent.comm_channel:
                                    agent.comm_channel.append((x, y, round(prob, 5)))
            elif (len(group) == 2):
                group_names = {agent.name for agent in group}
                all_agent_names = {agent.name for agent in world.agents}
                other_agents_names = all_agent_names - group_names
                if not other_agents_names:
                    continue
                print(f"当前分组不全面，调用LLM进行位置信息预测")
                reference_agent = random.choice(group)
                history_messages = []
                target_agents = []
                for other_agent_name in other_agents_names:
                    if other_agent_name in reference_agent.his_comm_channel:
                        for message in reference_agent.his_comm_channel[other_agent_name][-4:]:
                            history_messages.append(f"From {other_agent_name}: {message}")
                        target_agents.append(other_agent_name)
                    if other_agent_name in reference_agent.his_comm_channel_p_vel:
                        vel_history = reference_agent.his_comm_channel_p_vel[other_agent_name]
                        if vel_history:
                            last_vel = vel_history[-1]
                            history_messages.append(f"{other_agent_name} not in range. Last velocity: {last_vel}")
                    if other_agent_name in reference_agent.his_comm_channel_p_pos:
                        pos_history = reference_agent.his_comm_channel_p_pos[other_agent_name]
                        if pos_history:
                            last_pos = pos_history[-1]
                            history_messages.append(f"{other_agent_name} not in range. Last position: {last_pos}")
                if history_messages and target_agents:
                    result = self.predict_auv_trajectory(history_messages)
                    predicted_info = result.get("predicted_info", {})
                    velocity_info = result.get("velocity", {})
                    position_info = result.get("position", {})
                    for agent in group:
                        for raw_name in predicted_info.keys():
                            other_name = self.normalize_agent_name(raw_name)
                            if other_name in agent.his_comm_channel:
                                agent.his_comm_channel[other_name].extend(predicted_info[raw_name])
                                # single_agent.his_comm_channel[other_name].append(predicted_info[raw_name])
                            if other_name in agent.his_comm_channel_p_vel:
                                agent.his_comm_channel_p_vel[other_name].append(velocity_info.get(raw_name))
                            if other_name in agent.his_comm_channel_p_pos:
                                agent.his_comm_channel_p_pos[other_name].append(position_info.get(raw_name))
                        for target_name, predictions in predicted_info.items():
                            for x, y, prob in predictions:
                                # 限制 x 和 y 在 0 到 39 范围内
                                x = max(0, min(x, 39))
                                y = max(0, min(y, 39))
                                # 修改位置：避免重复添加
                                if (x, y, round(prob, 5)) not in agent.comm_channel:
                                    agent.comm_channel.append((x, y, round(prob, 5)))
            else:
                pass
            # 合并 comm_channel
            merged_comm_info = set()
            for agent in group:
                for (x, y, prob) in agent.comm_channel:
                    merged_comm_info.add((x, y, round(prob, 5)))
            for agent in group:
                agent.comm_channel = list(merged_comm_info)
            # 信息融合
            base_map = np.copy(group[0].probability_map)
            updates_dict = dict()
            for agent in group:
                for (x, y, prob) in agent.comm_channel:
                    if (x, y) not in updates_dict:
                        updates_dict[(x, y)] = []
                    updates_dict[(x, y)].append(prob)
            for (x, y), prob_list in updates_dict.items():
                if len(prob_list) >= 2:
                    base_map[x, y] = np.mean(prob_list)
                else:
                    base_map[x, y] = prob_list[0]
            total_prob = np.sum(base_map)
            if total_prob > 0:
                base_map /= total_prob
            else:
                print("Warning: Total probability is zero after fusion. Skipping normalization.")
            for agent in group:
                agent.probability_map = np.copy(base_map)
    def create_probability_map_with_squares(self, x_range, y_range, square_centers, square_side_length):
        x = np.linspace(x_range[0], x_range[1], num=40)
        y = np.linspace(y_range[0], y_range[1], num=40)
        X, Y = np.meshgrid(x, y)
        probability_map = np.zeros_like(X)
        total_cells = len(x) * len(y)
        inside_cells = 0
        self.visited_map = np.zeros_like(X, dtype=int)
        for center in square_centers:
            x_min = center[0] - square_side_length / 2
            x_max = center[0] + square_side_length / 2
            y_min = center[1] - square_side_length / 2
            y_max = center[1] + square_side_length / 2
            idx_x = np.where((x >= x_min) & (x <= x_max))[0]
            idx_y = np.where((y >= y_min) & (y <= y_max))[0]
            inside_cells += len(idx_x) * len(idx_y)
        outside_cells = total_cells - inside_cells
        p_outside = 1.0 / (inside_cells * 2 + outside_cells)
        p_inside = 2 * p_outside
        for i in range(len(x)):
            for j in range(len(y)):
                in_square = False
                for center in square_centers:
                    x_min = center[0] - square_side_length / 2
                    x_max = center[0] + square_side_length / 2
                    y_min = center[1] - square_side_length / 2
                    y_max = center[1] + square_side_length / 2
                    if x[i] >= x_min and x[i] <= x_max and y[j] >= y_min and y[j] <= y_max:
                        in_square = True
                        break
                if in_square:
                    probability_map[i, j] = p_inside
                else:
                    probability_map[i, j] = p_outside
        return probability_map

    def plot_probability_map(self, step, reward_dir=None, probability_map=None, agent_name=None):
        if reward_dir is None:
            reward_dir = self.reward_dir
        if probability_map is None:
            probability_map = agent.probability_map
        if probability_map is None or not probability_map.size:
            print("Probability map is not initialized or empty.")
            return
        plt.imshow(probability_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Probability')
        step //= 3
        title = f'Probability Map - Step {step}'
        if agent_name:
            title += f' - {agent_name}'
        plt.title(title)
        suffix = f"{agent_name}_" if agent_name else ""
        save_path = os.path.join(reward_dir, f'{suffix}probability_map_step_{step}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plt.savefig(save_path, format='png', bbox_inches='tight')
            csv_save_path = os.path.join(reward_dir, f'{suffix}probability_map_values_step_{step}.csv')
            with open(csv_save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['X', 'Y', 'Probability'])
                for i in range(probability_map.shape[0]):
                    for j in range(probability_map.shape[1]):
                        writer.writerow([i, j, probability_map[i, j]])
        except Exception as e:
            print(f"Failed to save image or data: {e}")
        finally:
            plt.close()

    def update_probability_map_with_multiplier(self, agent, world, fov, detection_prob, false_alarm_prob, observation, multiplier=10.0):
        for i in range(fov.shape[0]):
            for j in range(fov.shape[1]):
                if fov[i, j]:
                    if self.visited_map[i, j] < self.max_visits or agent.observation == 1:
                        prior = agent.probability_map[i, j]
                        prior = max(min(prior, 1 - 1e-6), 1e-6)
                        if agent.observation == 1:
                            likelihood_target = detection_prob
                            likelihood_no_target = false_alarm_prob
                        else:
                            likelihood_target = 1 - detection_prob
                            likelihood_no_target = 1 - false_alarm_prob
                        posterior = (likelihood_target * prior) / (
                                (likelihood_target * prior) + (likelihood_no_target * (1 - prior))
                        )
                        agent.probability_map[i, j] = posterior
                        if agent.observation == 1:
                            agent.probability_map[i, j] *= multiplier * 10
                        self.visited_map[i, j] += 1
        total_prob = np.sum(agent.probability_map)
        if total_prob > 0:
            agent.probability_map /= total_prob
        return agent.probability_map

    def check_agent_bounds(self, agent):
        map_min, map_max = -5, 5
        x, y = agent.state.p_pos[0], agent.state.p_pos[1]
        if x >= map_max or x <= map_min or y >= map_max or y <= map_min:
            return -20
        return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def calculate_target_discovery_reward(self, agent, world, fov):
        total_prob = 0
        count_cells_in_fov = 0
        for i in range(agent.probability_map.shape[0]):
            for j in range(agent.probability_map.shape[1]):
                if fov[i, j]:
                    total_prob += agent.probability_map[i, j]
                    count_cells_in_fov += 1
        if count_cells_in_fov > 0:
            reward = total_prob
        else:
            reward = 0
        return reward

    def get_fov(self, agent ,world):
        fov = np.zeros_like(agent.probability_map, dtype=bool)
        x_step = (5 - (-5)) / (agent.probability_map.shape[0] - 1)
        y_step = (5 - (-5)) / (agent.probability_map.shape[1] - 1)
        num = 0
        for i in range(agent.probability_map.shape[0]):
            for j in range(agent.probability_map.shape[1]):
                x = -5 + i * x_step
                y = -5 + j * y_step
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - [x,y])))
                if dist <= 0.3:
                    fov[i, j] = True
                    num +=1
        return fov

    def get_communication_groups(self, world, communication_range):
        groups = []
        grouped_agents = set()
        for agent in world.agents:
            if agent in grouped_agents:
                continue
            current_group = [agent]
            grouped_agents.add(agent)
            queue = [agent]
            while queue:
                current = queue.pop(0)
                for other in world.agents:
                    if other is not current and other not in grouped_agents:
                        distance = np.sqrt(np.sum(np.square(other.state.p_pos - current.state.p_pos)))
                        if distance < communication_range:
                            current_group.append(other)
                            grouped_agents.add(other)
                            queue.append(other)
            groups.append(current_group)
        return groups

    def accumulate_comm_channel(self, groups):
        for group in groups:
            for agent in group:
                for other_agent in group:
                    if other_agent is not agent:
                        if other_agent.name not in agent.his_comm_channel:
                            agent.his_comm_channel[other_agent.name] = []
                        temp_dict = {}
                        for (i, j, prob) in other_agent.comm_channel:
                            key = (i, j)
                            if key not in temp_dict or prob < temp_dict[key]:
                                temp_dict[key] = prob
                        seen = set()
                        filtered_channel = []
                        for (i, j, prob) in other_agent.comm_channel:
                            key = (i, j)
                            if key not in seen and prob == temp_dict[key]:
                                filtered_channel.append((i, j, prob))
                                seen.add(key)
                        agent.his_comm_channel[other_agent.name].extend(filtered_channel)
                        if other_agent.name not in agent.his_comm_channel_p_vel:
                            agent.his_comm_channel_p_vel[other_agent.name] = []
                        agent.his_comm_channel_p_vel[other_agent.name].append(other_agent.state.p_vel)
                        if other_agent.name not in agent.his_comm_channel_p_pos:
                            agent.his_comm_channel_p_pos[other_agent.name] = []
                        x, y = other_agent.state.p_pos
                        x_min, x_max = -5, 5
                        y_min, y_max = -5, 5
                        map_h, map_w = agent.probability_map.shape
                        x_step = (x_max - x_min) / (map_h - 1)
                        y_step = (y_max - y_min) / (map_w - 1)
                        x_index = int(round((x - x_min) / x_step))
                        y_index = int(round((y - y_min) / y_step))
                        x_index = min(max(0, x_index), map_h - 1)
                        y_index = min(max(0, y_index), map_w - 1)
                        agent.his_comm_channel_p_pos[other_agent.name].append((x_index, y_index))
    def truncate_his_comm_channel(self, agent, max_len=100):
        if len(agent.his_comm_channel) > max_len:
            agent.his_comm_channel = agent.his_comm_channel[-max_len:]

    def fuse_probability_maps_by_group1(self, groups):
        for group in groups:
            if len(group) <= 1:
                continue
            updates_dict = dict()
            for agent in group:
                for (x, y, prob) in agent.comm_channel:
                    if (x, y) not in updates_dict:
                        updates_dict[(x, y)] = []
                    updates_dict[(x, y)].append(prob)
            base_map = np.copy(group[0].probability_map)
            for (x, y), prob_list in updates_dict.items():
                base_map[x, y] = np.mean(prob_list)
            total_prob = np.sum(base_map)
            if total_prob > 0:
                base_map /= total_prob
            for agent in group:
                agent.probability_map = np.copy(base_map)

    def generate_real_target_positions(self):
        square_centers = [
            [-2.5, 2],
            [-2.5, 2],
            [-2.5, 2],
        ]
        square_centers1 = [
            [-2.5, 2],
        ]
        square_side_length = 3
        real_target_positions = []
        z = 0
        w = 0
        for center in square_centers1:
            x = round(random.uniform(center[0] - square_side_length / 2, center[0] + square_side_length / 2), 2)
            y = round(random.uniform(center[1] - square_side_length / 2, center[1] + square_side_length / 2), 2)
            x = round(x * 10) / 10 + 0.05 * (x % 1 != 0)
            y = round(y * 10) / 10 + 0.05 * (y % 1 != 0)
            z = x
            w = y
        for center in square_centers:
            real_target_positions.append([z, w])
        return real_target_positions
    def info(self, agent, world):
        comm_channel = []
        for (x, y, prob) in agent.comm_channel:
            comm_channel.append((x, y, prob))
        return np.concatenate([
            *comm_channel,
        ])

    def done(self, agent, world):
        return agent.tag

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks[0:1]:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        x, y = agent.state.p_pos
        x_index = int((x + 5) / (8 / (agent.probability_map.shape[0] - 1)))
        y_index = int((y + 5) / (8 / (agent.probability_map.shape[1] - 1)))
        x_index = min(max(0, x_index), agent.probability_map.shape[0] - 1)
        y_index = min(max(0, y_index), agent.probability_map.shape[1] - 1)
        probability_at_point = agent.probability_map[x_index, y_index]
        probability_at_point = np.array([probability_at_point], dtype=np.float32)
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([
            agent.state.p_vel,
            agent.state.p_pos,
            probability_at_point,
            *entity_pos,
            *other_pos,
        ])

    def reward(self, agent, world):
        dir = self.reward_dir
        self.numx += 1
        UWOC = 10
        rew = 0
        agent.c_energy +=1
        min_dists = 100
        tag = 0
        num = 1
        gl = 0
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        map_h, map_w = agent.probability_map.shape
        x_step = (x_max - x_min) / (map_h - 1)
        y_step = (y_max - y_min) / (map_w - 1)
        x = agent.state.p_pos[0]
        y = agent.state.p_pos[1]
        x_index = int(round((x - x_min) / x_step))
        y_index = int(round((y - y_min) / y_step))
        x_index = min(max(0, x_index), map_h - 1)
        y_index = min(max(0, y_index), map_w - 1)
        velocity = agent.state.p_vel
        velocity_str = f"[{velocity[0]:.4f}, {velocity[1]:.4f}]"
        for a in world.agents:
            if a is not agent:
                if a.tag == 0:
                    num += 1
                    agent.movable = False
                    agent.tag = 0
        if agent.tag == 0:
            rew = agent.max
            # if self.numx % 90000 == 0:
            if agent.name == "agent_2":
                for a in world.agents:
                    if a.name == "agent_0":
                        self.plot_probability_map(self.numx, dir, a.probability_map, agent_name="agent_0")
                    elif (a.name == "agent_1"):
                        self.plot_probability_map(self.numx, dir, a.probability_map, agent_name="agent_1")
                    else:
                        self.plot_probability_map(self.numx, dir, a.probability_map, agent_name="agent_2")
            return rew
        if agent.name == "agent_0":
            agent.ref = 1
            for lm in world.landmarks[agent.ref + 5: agent.ref + 6]:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
                for llm in world.landmarks[agent.ref-1: agent.ref]:
                    llm = np.sqrt(np.sum(np.square(agent.state.p_pos - llm.state.p_pos)))
                if dist < 0.5:
                    agent.observation1 = 1
                    print(f"Successfully discovered points of interest！{agent.name}已发现Target")
                    agent.movable = False
                    agent.tag = 0
                    rew += 100
                else:
                    agent.observation1 = 0
                fov = self.get_fov(agent, world)
                agent.probability_map = self.update_probability_map_with_multiplier(agent, world, fov, self.detection_prob, self.false_alarm_prob, agent.observation1, multiplier=10.0)
                rt = self.calculate_target_discovery_reward(agent, world, fov)
                rew += rt*1000
                rew -= llm*0.1
                probability_at_point1 = agent.probability_map[x_index, y_index]
                gl = probability_at_point1
                gl = round(gl, 5)
                existing_channel_map = {(i, j): p for (i, j, p) in agent.comm_channel}
                channel_map = {}
                triplet_log = (
                    f"position: {x_index , y_index}\n"
                    f"velocity: {velocity_str}\n"
                    "FOV Probability Triplets:\n"
                )
                for i in range(fov.shape[0]):
                    for j in range(fov.shape[1]):
                        if fov[i, j]:
                            prob = round(agent.probability_map[i, j], 5)
                            prob = max(prob, 0.00001)
                            key = (i, j)
                            triplet_log += f"({i}, {j}, {prob})\n"
                            if key in existing_channel_map:
                                prob = min(existing_channel_map[key], prob)
                            channel_map[key] = prob
                self.log_data(prompt=f"{agent.name} comm_channel", response_text=triplet_log)
                agent.comm_channel.extend([(i, j, prob) for (i, j), prob in channel_map.items()])
                unique_map = {}
                for i, j, prob in agent.comm_channel:
                    key = (i, j)
                    if key in unique_map:
                        unique_map[key] = min(unique_map[key], prob)
                    else:
                        unique_map[key] = prob
                agent.comm_channel = [(i, j, prob) for (i, j), prob in unique_map.items()]

        elif(agent.name == "agent_1"):
            agent.ref = 1
            for lm in world.landmarks[agent.ref+5:agent.ref+6]:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
                for llm in world.landmarks[agent.ref-1: agent.ref]:
                    llm = np.sqrt(np.sum(np.square(agent.state.p_pos - llm.state.p_pos)))
                if dist < 0.5:
                    agent.observation1 = 1
                    print(f"Successfully discovered points of interest！{agent.name}已发现Target")
                    agent.movable = False
                    agent.tag = 0
                    rew += 100
                else:
                    agent.observation1 = 0
                fov = self.get_fov(agent, world)
                agent.probability_map = self.update_probability_map_with_multiplier(agent, world , fov, self.detection_prob, self.false_alarm_prob, agent.observation1, multiplier=10.0)
                rt = self.calculate_target_discovery_reward(agent, world ,fov)
                rew += rt*1000
                rew -= llm*0.1
                probability_at_point1 = agent.probability_map[x_index, y_index]
                gl = probability_at_point1
                gl = round(gl, 5)
                existing_channel_map = {(i, j): p for (i, j, p) in agent.comm_channel}
                channel_map = {}
                triplet_log = (
                    f"position: {x_index, y_index}\n"
                    f"velocity: {velocity_str}\n"
                    "FOV Probability Triplets:\n"
                )
                for i in range(fov.shape[0]):
                    for j in range(fov.shape[1]):
                        if fov[i, j]:
                            prob = round(agent.probability_map[i, j], 5)
                            prob = max(prob, 0.00001)
                            key = (i, j)
                            triplet_log += f"({i}, {j}, {prob})\n"
                            if key in existing_channel_map:
                                prob = min(existing_channel_map[key], prob)
                            channel_map[key] = prob
                self.log_data(prompt=f"{agent.name} comm_channel", response_text=triplet_log)
                agent.comm_channel.extend([(i, j, prob) for (i, j), prob in channel_map.items()])
                unique_map = {}
                for i, j, prob in agent.comm_channel:
                    key = (i, j)
                    if key in unique_map:
                        unique_map[key] = min(unique_map[key], prob)
                    else:
                        unique_map[key] = prob
                agent.comm_channel = [(i, j, prob) for (i, j), prob in unique_map.items()]
        elif(agent.name == "agent_2"):
            agent.ref = 1
            for lm in world.landmarks[agent.ref+5:agent.ref+6]:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
                for llm in world.landmarks[agent.ref-1: agent.ref]:
                    llm = np.sqrt(np.sum(np.square(agent.state.p_pos - llm.state.p_pos)))
                if dist < 0.5:
                    agent.observation1 = 1
                    print(f"Successfully discovered points of interest！{agent.name}已发现Target")
                    agent.movable = False
                    agent.tag = 0
                    rew += 100
                else:
                    agent.observation1 = 0
                fov = self.get_fov(agent, world)
                agent.probability_map = self.update_probability_map_with_multiplier(agent, world, fov, self.detection_prob, self.false_alarm_prob, agent.observation1, multiplier=10.0)
                rt = self.calculate_target_discovery_reward(agent, world, fov)
                rew += rt*1000
                rew -= llm*0.1
                probability_at_point1 = agent.probability_map[x_index, y_index]
                gl = probability_at_point1
                existing_channel_map = {(i, j): p for (i, j, p) in agent.comm_channel}
                channel_map = {}
                triplet_log = (
                    f"position: {x_index, y_index}\n"
                    f"velocity: {velocity_str}\n"
                    "FOV Probability Triplets:\n"
                )
                for i in range(fov.shape[0]):
                    for j in range(fov.shape[1]):
                        if fov[i, j]:
                            prob = round(agent.probability_map[i, j], 5)
                            prob = max(prob, 0.00001)
                            key = (i, j)
                            triplet_log += f"({i}, {j}, {prob})\n"
                            if key in existing_channel_map:
                                prob = min(existing_channel_map[key], prob)
                            channel_map[key] = prob
                self.log_data(prompt=f"{agent.name} comm_channel", response_text=triplet_log)
                agent.comm_channel.extend([(i, j, prob) for (i, j), prob in channel_map.items()])
                unique_map = {}
                for i, j, prob in agent.comm_channel:
                    key = (i, j)
                    if key in unique_map:
                        unique_map[key] = min(unique_map[key], prob)
                    else:
                        unique_map[key] = prob
                agent.comm_channel = [(i, j, prob) for (i, j), prob in unique_map.items()]
        if agent.collide:
            for a in world.agents:
                if a.tag != 0:
                    if a is not agent and self.is_collision(a, agent):
                        rew -= 20
        bounds_reward = self.check_agent_bounds(agent)
        rew += bounds_reward
        agent.max = rew
        agent.probability_at_point1 = gl
        if self.numx % 3 == 0:
            groups = self.get_communication_groups(world, UWOC)
            self.accumulate_comm_channel(groups)
            self.fuse_probability_maps_by_group(groups)
        if self.numx % 600000 == 0:
            for a in world.agents:
                if a.name == "agent_0":
                    self.plot_probability_map(self.numx, dir, a.probability_map, agent_name="agent_0")
                elif(a.name == "agent_1"):
                    self.plot_probability_map(self.numx, dir, a.probability_map, agent_name="agent_1")
                else:
                    self.plot_probability_map(self.numx, dir, a.probability_map, agent_name="agent_2")
        return rew