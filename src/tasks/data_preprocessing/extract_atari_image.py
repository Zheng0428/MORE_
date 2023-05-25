import sys
sys.path.append("./src")
sys.path.append("./src/tasks/data_preprocessing")
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LxmertTokenizerFast
from lxrt_dataload import LXMTDataLoad
from tasks.data_preprocessing.utils import extend_tensor
from fasterrcnn import FasterRCNN_Visual_Feats
import csv, os, sys, time
from fixed_replay_buffer import FixedReplayBuffer
class LoadData(nn.Module):
    def __init__(self, dataset_path, max_length, tokenizer_config = 'unc-nlp/lxmert-base-uncased', reward_with_timestep=True, device = 'cpu'):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = LxmertTokenizerFast.from_pretrained(tokenizer_config)
        self.faster_r_cnn = FasterRCNN_Visual_Feats(device=self.device)
        self.reward_with_timestep = reward_with_timestep
        self.observations = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.instructions = []
        self.episode_idxs = None
        self.episode_lengths = None
        self.rtg = None
        # init lxrt model
        self.lxrt_feature = None
        self.lxrt_dataload = LXMTDataLoad(64, self.device)
        self.outfile = './data/atari_imgfeat/'
        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

    def get_non_zero_idx(self, env):
        end_idxs_lst = torch.nonzero(torch.as_tensor(self.trajectories[env]['dones']))
        if len(end_idxs_lst) == 0:
            end_idxs_lst = torch.as_tensor([self.max_length * (i+1) - 1 for i in range(3)])
        return end_idxs_lst

    def get_episode_infos(self, rewards):
        episode_idx = torch.zeros_like(rewards)
        episode_lengths = torch.zeros_like(rewards)
        end_idxs_lst = len(rewards)
        episode_idx.fill_(end_idxs_lst-1)
        episode_lengths.fill_(end_idxs_lst)
        return episode_idx.to(torch.int64), episode_lengths.to(torch.int64)
    
    def get_rtg(self, rewards):
        rtg = np.zeros_like(rewards)
        episode_length = len(rewards)
        for j in range(episode_length - 1, -1, -1): # start from i-1
            rtg_j = rewards[j : episode_length]
            rtg[episode_length - 1 - j] = sum(rtg_j)
        return rtg



    def split_data(self, data, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        assert train_ratio + valid_ratio + test_ratio == 1, "比例之和必须等于1"

        # 计算数据总量
        total_size = len(data)
        
        # 生成一个乱序索引数组
        shuffled_indices = np.random.permutation(total_size)

        # 计算各个子集的大小
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)

        # 根据乱序索引划分数据集
        train_data = [data[i] for i in shuffled_indices[:train_size]]
        valid_data = [data[i] for i in shuffled_indices[train_size:train_size + valid_size]]
        test_data = [data[i] for i in shuffled_indices[train_size + valid_size:]]

        return train_data, valid_data, test_data
    
    def lxrt_out(self, action, state):
        pass

    def forward(self):
        if os.path.exists(self.outfile +'atari.pt'):
            data = torch.load(self.outfile +'atari.pt')
            print("Already have %d data from split(s) %s." % (len(data[0]), 'atari'))
            size = len(data[0])
        else:
            size = 0
        data = None
        flag = 0
        sum = 0
        data_dir_prefix = '/home/zhangge/ZTY_Adam/decision-transformer/atari/dqn_replay/'
        game = 'Breakout'
        num_buffers = 50
        transitions_per_buffer = np.zeros(50, dtype=int)
        buffer_num = buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs/',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            actions = []
            states = []
            rewards = []
            trajectories_to_load = 10
            times = time.time()
            b = len(states)
            lens = 0
            while not done:
                state, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i]) 
                if len(states) < 1000:
                    states += [state[0]]# (1, 84, 84, 1) --> (84, 84, 1)
                    actions += [ac[0]]
                    rewards += [ret[0]]
                if terminal[0]:
                    print("The length of the %d track is %d ." % ((11-trajectories_to_load),lens))
                    if (11-trajectories_to_load) > size: 
                        states = [torch.from_numpy(arr) for arr in states]
                        states = torch.stack(states)
                        actions = torch.as_tensor(actions)
                        episode_idxs, episode_lengths = self.get_episode_infos(torch.as_tensor(rewards))
                        visual_feats, visual_pos = self.faster_r_cnn([img for img in states])
                        actions, visual_feats, visual_pos = actions.to(self.device), visual_feats.to(self.device), visual_pos.to(self.device)
                        if os.path.exists(self.outfile +'atari.pt'):
                            data = torch.load(self.outfile +'atari.pt')
                            print("Load %d data from split(s) %s." % (len(data[0]), 'atari'))
                        else:
                            data = [None, None, None, None, None, None]
                        self.lxrt_feature = extend_tensor(data[0],(self.lxrt_dataload(actions, visual_feats, visual_pos)).to('cpu'))
                        self.rtg = extend_tensor(data[1], torch.as_tensor(self.get_rtg(rewards)))
                        self.rewards = extend_tensor(data[2], torch.as_tensor(rewards))
                        self.actions = extend_tensor(data[3],actions)
                        self.episode_idxs = extend_tensor(data[4], episode_idxs)
                        self.episode_lengths = extend_tensor(data[5], episode_lengths)
                        data = None
                        print ("time:"+ str(time.time()-times))
                        times = time.time()
                        # self.actions = extend_tensor(self.actions,actions)
                        # self.rtg = extend_tensor(self.rtg, torch.as_tensor(self.get_rtg(rewards)))
                        # self.rewards = extend_tensor(self.rewards, torch.as_tensor(rewards))
                        # episode_idxs, episode_lengths = self.get_episode_infos(torch.as_tensor(rewards))
                        # self.episode_idxs = extend_tensor(self.episode_idxs, episode_idxs)
                        # self.episode_lengths = extend_tensor(self.episode_lengths, episode_lengths)
                        # visual_feats, visual_pos = self.faster_r_cnn([img for img in states])
                        # actions, visual_feats, visual_pos = actions.to(self.device), visual_feats.to(self.device), visual_pos.to(self.device)
                        # self.lxrt_feature = extend_tensor(self.lxrt_feature,(self.lxrt_dataload(actions, visual_feats, visual_pos)).to('cpu'))

                        torch.save([self.lxrt_feature,
                            self.rtg,
                            self.rewards,
                            self.actions,
                            self.episode_idxs,
                            self.episode_lengths]
                        ,self.outfile +'atari.pt')
                        print ('done')
                    lens = 0
                    actions = []
                    states = []
                    rewards = []
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                i += 1
                lens += 1
                if i % 5000 == 0:
                    a = 1
                # visual_feats, visual_pos = self.faster_r_cnn(states)
                # action, visual_feats, visual_pos = ac.to(self.device), visual_feats.to(self.device), visual_pos.to(self.device)
                # self.lxrt_feature = extend_tensor(self.lxrt_feature,(self.lxrt_dataload(action, visual_feats, visual_pos)).to('cpu'))
                
                a = 1
        # data = torch.load(self.outfile + 'train10.pt')
        # self.lxrt_feature = extend_tensor(data[0] , self.lxrt_feature)
        # self.rtg = extend_tensor(data[1] , self.rtg)
        # self.rewards = extend_tensor(data[2] , self.rewards)
        # self.actions = extend_tensor(data[3] , self.actions)
        # self.instructions.extend(data[4])
        # self.episode_idxs = extend_tensor(data[5] , self.episode_idxs)
        # self.episode_lengths = extend_tensor(data[6] , self.episode_lengths)
        print ('finish')
        print (sum)
        # data = [self.lxrt_feature,
        #         self.rtg,
        #         self.rewards,
        #         self.actions,
        #         self.instructions,
        #         self.episode_idxs,
        #         self.episode_lengths]
        # if os.path.exists(self.outfile):
        #     torch.save(data,self.outfile +'train1.pt')


        #np.savetxt('./obj/model.csv',model.encode().detach().numpy(),fmt='%.2f',delimiter=',')
        return self.outfile

def split_data(data, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    assert train_ratio + valid_ratio + test_ratio == 1, "比例之和必须等于1"

    # 计算数据总量
    total_size = len(data[0])
    
    # 生成一个乱序索引数组
    shuffled_indices = np.random.permutation(total_size)

    # 计算各个子集的大小
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    train_data, valid_data, test_data = [], [], []
    # 根据乱序索引划分数据集
    for i in range(7):
        train_data[i] = shuffled_indices[i][:train_size]
        valid_data[i] = shuffled_indices[i][train_size:train_size + valid_size]
        test_data[i] = shuffled_indices[i][train_size + valid_size:]
    return train_data, valid_data, test_data

def fuc2():
    outfile = './data/minigrid_imgfeat/'
    data = torch.load(outfile +'all.pt')
    train_data, valid_data, test_data = split_data(data, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)
    if os.path.exists(outfile):
        torch.save(train_data,outfile +'train.pt')
        torch.save(valid_data,outfile +'valid.pt')
        torch.save(test_data, outfile +'test.pt')
        print ('finish')

def fuc():
    outfile = './data/minigrid_imgfeat/'
    data1 = torch.load(outfile +'train1.pt')
    a = len(data1[0])
    data2 = torch.load(outfile +'train2.pt')
    a = a + len(data2[0])
    result = []
    for i in range(7):
        if i == 4:
            concatenated = []
            concatenated.extend(data1[i])
            concatenated.extend(data2[i])
        else:
            concatenated = torch.cat((data1[i], data2[i]),dim = 0)
        result.append(concatenated)
    if os.path.exists(outfile):
        torch.save(result,outfile +'train.pt')
        print ('finish')


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:

    # Load image ids, need modification for new datasets.
    # fuc()
    # fuc2()
    data = LoadData(device = 'cuda:0', max_length = 100, dataset_path='/home/zhangge/ZTY_Adam/MORE/data/more/minigrid_traj.pkl')
    a = data()
    # Generate TSV files, noramlly do not need to modify

