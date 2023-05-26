import sys
sys.path.append("/home/biao/MORE_/src")
import gym
import torch
import torch.nn as nn
import numpy as np
import atari_py
import cv2
from tasks.action import ActionNet

from tasks.data_preprocessing.fasterrcnn import FasterRCNN_Visual_Feats
from tasks.data_preprocessing.lxrt_dataload import LXMTDataLoad




class ValidAtari:
    def __init__(self, device, model_outfile='/home/biao/MORE_data/model/'):
        self.device = device
        self.faster_r_cnn = FasterRCNN_Visual_Feats(device=self.device)
        self.lxrt_dataload = LXMTDataLoad(64, self.device)
        self.outfile = model_outfile
        self.itr_model = ActionNet()
        self.itr_model.load_state_dict(torch.load(self.outfile + 'classifier.pth'))
        self.itr_model = self.itr_model.to(self.device)

    def preprocess_state(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # 将状态转换为灰度
        state = cv2.resize(state, (84, 84))  # 调整状态的大小
        state = state / 255.0  # 归一化状态
        state = np.expand_dims(state, axis=0)  # 增加通道维度
        return state

    def intermediate(self, model, data, device):
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
        return predicted

    def lxrt_out(self, state, action):
        action = torch.tensor(action).unsqueeze(0)
        state = state.permute(1, 2, 0)
        visual_feats, visual_pos = self.faster_r_cnn(state.unsqueeze(0))
        action, visual_feats, visual_pos = action.to(self.device), visual_feats.to(self.device), visual_pos.to(self.device)
        lxmert_out = self.lxrt_dataload(action, visual_feats, visual_pos)
        return lxmert_out

    def __call__(self, model):       
        
        # itr_model = torch.load(outfile + 'classifier.pth')
        # model.load_state_dict(torch.load("dqn_model.pth"))
        # model = model.to(device)
        # model.eval()
        # 创建游戏环境
        env = gym.make('Breakout-v0')

        T_rewards = []
        # 对模型进行评估
        num_episodes = 10
        ret = 90
        done = True
        for i in range(num_episodes):
            rtgs = [ret]
            state = env.reset()
            state = self.preprocess_state(state[0])  # 预处理状态
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            lxmert_out = self.lxrt_out(state[0], 0)
            with torch.no_grad():
                output = model(lxmert_out=lxmert_out.unsqueeze(0),           
                    rtg=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0), 
                    traj_mask=torch.zeros((1, 1), dtype=torch.int64).to(self.device),
                    timesteps=torch.zeros((1, 1), dtype=torch.int64).to(self.device)
                )
            sampled_action = self.intermediate(self.itr_model, output.logits, self.device)
            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                all_states = torch.cat([all_states, state], dim=0)
                rtgs += [rtgs[-1] - reward]
                with torch.no_grad():
                    output = model(all_states.unsqueeze(0)
                                )
                sampled_action = self.intermediate(self.itr_model, output, self.device)
                next_state, reward, done, _ = env.step(sampled_action.item())
                next_state = self.preprocess_state(next_state)  # 预处理状态
                next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)

                state = next_state
                if done:
                    break



        # 计算平均得分
        mean_score = np.mean(T_rewards)
        std_score = np.std(T_rewards)
        print(f"Mean score over {num_episodes} episodes: {mean_score} (std: {std_score})")
