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
    def __init__(self, dataset_path, max_length=1000, tokenizer_config = 'unc-nlp/lxmert-base-uncased', reward_with_timestep=False, device = 'cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = LxmertTokenizerFast.from_pretrained(tokenizer_config)
        self.faster_r_cnn = FasterRCNN_Visual_Feats(device=self.device)
        self.reward_with_timestep = reward_with_timestep
        self.max_length = max_length
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
    def forward(self):
        for env in self.trajectories:
            print(env)
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

        if os.path.exists(self.outfile):
            torch.save([self.lxrt_feature,
                        self.rtg,
                        self.rewards,
                        self.actions,
                        self.instructions,
                        self.episode_idxs,
                        self.episode_lengths],self.outfile +'train.pt')


        #np.savetxt('./obj/model.csv',model.encode().detach().numpy(),fmt='%.2f',delimiter=',')
        return self.outfile




if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:

    # Load image ids, need modification for new datasets.
    data = LoadData(device = 'cuda:0',dataset_path='/home/zhangge/ZTY_Adam/MORE/data/more/minigrid_traj.pkl')
    a = data()
    # Generate TSV files, noramlly do not need to modify

