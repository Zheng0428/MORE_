import sys
sys.path.append("/home/biao/MORE_/src")
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import torch
import torch.nn as nn
import numpy as np
import atari_py
import torch.nn.functional as F
import cv2
from tasks.action import ActionNet

from tasks.data_preprocessing.fasterrcnn import FasterRCNN_Visual_Feats
from tasks.data_preprocessing.lxrt_dataload import LXMTDataLoad
MAX_LENGTH = 1000

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
    
    def preprocess_observation(self, obs):
        img = obs[34:194:2, ::2]  # 裁剪和下采样
        img = img.mean(axis=2)  # 转换为灰度
        img[img==144] = 0  # 清除背景
        img[img==109] = 0  # 清除背景
        img[img!=0] = 1  # 显示所有物体为白色
        return np.array([img.reshape(80, 80, 1)])  # 添加一维用于批处理

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
        env = AtariPreprocessing(gym.make('BreakoutNoFrameskip-v0'))
        T_rewards = []
        # 对模型进行评估
        num_episodes = 10
        ret = 0
        done = True
        timesteps = torch.arange(start=0, end=MAX_LENGTH, step=1).unsqueeze(0)
        for i in range(num_episodes):
            j = 1
            rtgs = [ret]
            state = env.reset()
            # state = self.preprocess_observation(state[0])  # 预处理状态
            state = torch.from_numpy(state[0]).float().unsqueeze(0).to(self.device)
            lxmert_out = self.lxrt_out(state, 0)
            with torch.no_grad():
                output = model(lxmert_out=F.pad(lxmert_out.unsqueeze(0), (0, 0, 0, 0, 0, MAX_LENGTH - j), "constant", 0),           
                        rtg=torch.cat((torch.tensor(rtgs).unsqueeze(0),torch.zeros(1, MAX_LENGTH-j)), dim=1).to(self.device), 
                        traj_mask=torch.cat((torch.ones(1, 1), torch.zeros(1, MAX_LENGTH - j)), dim=1).to(self.device),
                        timesteps=timesteps.to(self.device)
                    )
                # output = model(lxmert_out=lxmert_out.unsqueeze(0),           
                #     rtg=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0), 
                #     traj_mask=torch.ones((1, 1), dtype=torch.int64).to(self.device),
                #     timesteps=torch.zeros((1, 1), dtype=torch.int64).to(self.device)
                # )
            # a = output.hidden_states[-1][:, 6, :]
            sampled_action = self.intermediate(self.itr_model, output.hidden_states[-1][:, j:j+1, :], self.device)
            all_states = lxmert_out
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[-1]
                actions += [sampled_action]
                state, reward, done_terminated, done_truncated, info = env.step(action)
                if reward != 0.0:
                    a = 1
                if done_terminated:
                    done = True
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                # state = self.preprocess_observation(state)  # 预处理状态
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                lxmert_out = self.lxrt_out(state, action)

                all_states = torch.cat([all_states, lxmert_out], dim=0)
                rtgs += [rtgs[-1] + int(reward)]
                padding = (0, 0, 0, 0, 0, MAX_LENGTH - j)
                with torch.no_grad():
                    output = model(lxmert_out=F.pad(all_states.unsqueeze(0), padding, "constant", 0),           
                        rtg=torch.cat((torch.tensor(rtgs).unsqueeze(0),torch.zeros(1, MAX_LENGTH-j)), dim=1).to(self.device), 
                        traj_mask=torch.cat((torch.ones(1, j), torch.zeros(1, MAX_LENGTH - j)), dim=1).to(self.device),
                        timesteps=timesteps.to(self.device)
                    )
                    # output = model(lxmert_out=all_states.unsqueeze(0),           
                    #     rtg=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0), 
                    #     traj_mask=torch.ones((1, j), dtype=torch.int64).to(self.device),
                    #     timesteps=torch.arange(j, dtype=torch.int64).reshape(1,j).to(self.device)
                    # )

                sampled_action = self.intermediate(self.itr_model, output.hidden_states[-1][:, j:j+1, :], self.device)

                # next_state, reward, done, _ = env.step(sampled_action.item())
                # next_state = self.preprocess_state(next_state)  # 预处理状态
                # next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                # state = next_state

        env.close()
        # 计算平均得分
        mean_score = np.mean(T_rewards)
        std_score = np.std(T_rewards)
        print(f"Mean score over {num_episodes} episodes: {mean_score} (std: {std_score})")
