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
import csv, os, sys
class LoadData(nn.Module):
    def __init__(self, dataset_path, max_length, tokenizer_config = 'unc-nlp/lxmert-base-uncased', reward_with_timestep=False, device = 'cpu'):
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
        self.outfile = './data/minigrid_imgfeat/'
        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

    def get_non_zero_idx(self, env):
        end_idxs_lst = torch.nonzero(torch.as_tensor(self.trajectories[env]['dones']))
        if len(end_idxs_lst) == 0:
            end_idxs_lst = torch.as_tensor([self.max_length * (i+1) - 1 for i in range(3)])
        return end_idxs_lst

    def get_episode_infos(self, env):
        episode_idx = None
        episode_lengths = None
        end_idxs_lst = self.get_non_zero_idx(env)
        for val in end_idxs_lst:
            if episode_idx is None:
                episode_length = val + 1
            else:
                episode_length = val + 1 - episode_idx.shape[0]
            episode_idx = extend_tensor(episode_idx, torch.as_tensor([self.observations.shape[0] + val - len(self.trajectories[env]['dones']) for i in range(episode_length)]))
            episode_lengths = extend_tensor(episode_lengths, torch.as_tensor([episode_length for i in range(episode_length)]))
        return episode_idx, episode_lengths

    def get_rtg(self, env):
        rewards = self.trajectories[env]['rewards']
        rtg = None
        end_idxs_lst = self.get_non_zero_idx(env)
        for val in end_idxs_lst:
            if rtg is None:
                episode_length = val + 1
            else:
                episode_length = val + 1 - rtg.shape[0]
            curr_episode = [rewards[val] for i in range(episode_length)]
            if self.reward_with_timestep:
                reward = rewards[val]
                curr_episode[:] = [reward / episode_length * i for i in range(curr_episode)]
            rtg = extend_tensor(rtg, torch.as_tensor(curr_episode))
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
    
    def forward(self):
        flag = 0
        val=[
            'LavaCrossingS9N2-v0',  
            'SimpleCrossingS9N3-v0', 
            'DistShift1-v0',
            'DoorKey-5x5-v0',
            'Dynamic-Obstacles-8x8-v0',
            'Empty-Random-6x6-v0',
            'Empty-16x16-v0',
            'Fetch-6x6-N2-v0',
            'GoToDoor-5x5-v0',
            'KeyCorridorS4R3-v0',
            'LavaGapS5-v0',
            'MemoryS17Random-v0',
            'MultiRoom-N4-S5-v0',
            'PutNear-6x6-N2-v0',
            'RedBlueDoors-6x6-v0',
        ]
        sum = 0
        for env in self.trajectories:
            print(env)
            if env == 'Fetch-6x6-N2-v0':
                break
            if env == 'Empty-6x6-v0':
                flag = 1
            if flag == 0:
                continue
            if env in val:
                continue
            sum += len(self.trajectories[env]['observations'])
            self.observations = extend_tensor(self.observations, torch.as_tensor(self.trajectories[env]['observations']))
            self.instructions.extend(self.trajectories[env]['instructions'])
            self.actions = extend_tensor(self.actions,torch.as_tensor(self.trajectories[env]['actions']))
            self.rewards = extend_tensor(self.rewards, torch.as_tensor(self.trajectories[env]['rewards']))
            self.dones = extend_tensor(self.dones, torch.as_tensor(self.trajectories[env]['dones']))
            episode_idxs, episode_lengths = self.get_episode_infos(env)
            self.episode_idxs = extend_tensor(self.episode_idxs, episode_idxs)
            self.episode_lengths = extend_tensor(self.episode_lengths, episode_lengths)
            self.rtg = extend_tensor(self.rtg, self.get_rtg(env))
            # lxmert output
            action = torch.as_tensor(self.trajectories[env]['actions'])
            visual_feats, visual_pos = self.faster_r_cnn([img for img in self.observations])
            action, visual_feats, visual_pos = action.to(self.device), visual_feats.to(self.device), visual_pos.to(self.device)
            self.lxrt_feature = extend_tensor(self.lxrt_feature,(self.lxrt_dataload(action, visual_feats, visual_pos)).to('cpu'))
            # torch.save([self.lxrt_feature,
            #             self.rtg,
            #             self.rewards,
            #             self.actions,
            #             self.instructions,
            #             self.episode_idxs,
            #             self.episode_lengths]
            #            ,self.outfile +'train20.pt')
        data = torch.load(self.outfile + 'train10.pt')
        self.lxrt_feature = extend_tensor(data[0] , self.lxrt_feature)
        self.rtg = extend_tensor(data[1] , self.rtg)
        self.rewards = extend_tensor(data[2] , self.rewards)
        self.actions = extend_tensor(data[3] , self.actions)
        self.instructions.extend(data[4])
        self.episode_idxs = extend_tensor(data[5] , self.episode_idxs)
        self.episode_lengths = extend_tensor(data[6] , self.episode_lengths)
        print ('finish')
        print (sum)
        data = [self.lxrt_feature,
                self.rtg,
                self.rewards,
                self.actions,
                self.instructions,
                self.episode_idxs,
                self.episode_lengths]
        if os.path.exists(self.outfile):
            torch.save(data,self.outfile +'train1.pt')


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
    fuc()
    # fuc2()
    # data = LoadData(device = 'cuda:1', max_length = 1000, dataset_path='/home/zhangge/ZTY_Adam/MORE/data/more/minigrid_traj.pkl')
    # a = data()
    # Generate TSV files, noramlly do not need to modify

