import gym
import torch
import torch.nn as nn
import numpy as np
import atari_py
import cv2

# 定义模型，这需要与你的原始模型一致
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(56 * 768, 128) # 扁平化后的输入大小为 56*768
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 四种动作，所以最后一层的输出大小为4

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 对输入张量进行扁平化处理
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output



def preprocess_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # 将状态转换为灰度
    state = cv2.resize(state, (84, 84))  # 调整状态的大小
    state = state / 255.0  # 归一化状态
    state = np.expand_dims(state, axis=0)  # 增加通道维度
    return state

def intermediate(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
    return predicted

def lxrt_out(state, action):
    lxmert_out = 0
    return lxmert_out

outfile = '/home/zhangge/ZTY_Adam/MORE1_/data/atari_imgfeat/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TestNet()
itr_model = torch.load(outfile + 'classifier.pth')
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
    state = preprocess_state(state)  # 预处理状态
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    lxmert_out = lxrt_out(state, 0)
    with torch.no_grad():
        output = model(lxmert_out,
            rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device)
        )
    sampled_action = intermediate(itr_model, output, device)
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

        state = state.unsqueeze(0).unsqueeze(0).to(device)
        all_states = torch.cat([all_states, state], dim=0)
        rtgs += [rtgs[-1] - reward]
        with torch.no_grad():
            output = model(all_states.unsqueeze(0)
                           )
        sampled_action = intermediate(itr_model, output, device)
        next_state, reward, done, _ = env.step(sampled_action.item())
        next_state = preprocess_state(next_state)  # 预处理状态
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

        state = next_state
        if done:
            break



# 计算平均得分
mean_score = np.mean(T_rewards)
std_score = np.std(T_rewards)
print(f"Mean score over {num_episodes} episodes: {mean_score} (std: {std_score})")
