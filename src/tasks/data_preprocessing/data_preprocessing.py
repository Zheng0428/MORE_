import sys
sys.path.append("./src")
sys.path.append("./src/tasks/data_preprocessing")
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LxmertTokenizerFast
from lxrt_dataload import LXMTDataLoad
from tasks.data_preprocessing.utils import extend_tensor
from fasterrcnn import FasterRCNN_Visual_Feats

'''
Usage:

traj_dataset = MiniGridDataset(dataset_path='minigrid.pkl')

traj_data_loader = DataLoader(traj_dataset,
						batch_size=batch_size,
						shuffle=True,
						pin_memory=True,
						drop_last=True)

data_iter = iter(traj_data_loader)

example:

timesteps, states, actions, returns_to_go, traj_mask, instructions_input_ids, instructions_token_type_ids, instructions_attention_mask, visual_feats, visual_pos = next(data_iter)

context_length -> 1000 (long enough to contain all episodes, most will be shorter with padding)

timesteps -> tensor[batch_size, context_length]
states -> tensor[batch_size, context_length, 56, 56, 3]
actions -> tensor[batch_size, context_length]
returns_to_go -> tensor[batch_size, context_length]
traj_mask -> tensor[batch_size, context_length]

** added: tokenized instructions, set max_length of instructions to 32 **

instructions_input_ids -> [batch_size, context_length, 32]
instructions_token_type_ids -> [batch_size, context_length, 32]
instructions_attention_mask -> [batch_size, context_length, 32]

** added: visual feats and visual pos from faster r-cnn **

visual_feats -> [batch_size, context_length, 36, 2048]
visual_pos -> [batch_size, context_length, 36, 4]
'''

class MiniGridDataset(Dataset):
    def __init__(self, splits, path, max_length=1000, device = 'cpu'):
        self.device = device
        self.splits = splits.split(',')
        path = path + ('%s.pt' % self.splits[0])
        self.max_length = max_length
        path = '/home/biao/MORE_data/atari_data/atari.pt'
        # load dataset
        if os.path.exists(path):
            self.data = torch.load(path)
            print("Load %d data from split(s) %s." % (len(self.data[0]), splits))
        self.lxrt_feature = self.data[0]
        self.rtg = self.data[1]
        self.rewards = self.data[2]
        self.actions = self.data[3].cpu()
        # self.instructions = self.data[4]
        # self.episode_idxs = self.data[5]
        # self.episode_lengths = self.data[6]

        self.episode_idxs = self.data[4]
        self.episode_lengths = self.data[5]

        '''
        Discounts to go -> option of having 1/0 for all return-to-go for an episode,
        or have return-to-go decrease with each timestep
        '''

    def __len__(self):
        return len(self.rtg)

    def __getitem__(self, index):
        '''
        returns the rest of the given episode from the indexed time step
        '''
        episode_length = self.episode_lengths[index]
        episode_end_idx = self.episode_idxs[index]
        padding_length = self.max_length - episode_length
        if padding_length < 0:
            actions = self.actions[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
            actions = actions.numpy()
            actions = actions[:self.max_length]

            rtg = self.rtg[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
            rtg = rtg.numpy()
            rtg = rtg[:self.max_length]
            timesteps = torch.arange(start=0, end=self.max_length, step=1)

            traj_mask = torch.ones(self.max_length, dtype=torch.long)

            # lxrt ouput part
            lxrt_feature = self.lxrt_feature[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
            lxrt_feature = lxrt_feature.numpy()
            lxrt_feature = lxrt_feature[:self.max_length]
        else:
            actions = self.actions[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
            actions = actions.numpy()
            actions = np.concatenate((actions,
                                    np.zeros(([padding_length] + list(actions.shape[1:])),
                                    dtype=actions.dtype)),
                                    axis=0)

            rtg = self.rtg[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
            rtg = rtg.numpy()
            rtg = np.concatenate((rtg,
                                np.zeros(([padding_length] + list(rtg.shape[1:])),
                                dtype=rtg.dtype)),
                                axis=0)

            # instructions = self.instructions[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
            # instructions = self.tokenizer(instructions, return_tensors="pt", max_length=32, padding='max_length')
            # instructions_input_ids = torch.cat([instructions['input_ids'],
            #                     torch.zeros(([padding_length] + list(instructions['input_ids'].shape[1:])),
            #                     dtype=instructions['input_ids'].dtype)],
            #                     dim=0)
            # instructions_token_type_ids = torch.cat([instructions['token_type_ids'],
            #                     torch.zeros(([padding_length] + list(instructions['token_type_ids'].shape[1:])),
            #                     dtype=instructions['token_type_ids'].dtype)],
            #                     dim=0)
            # instructions_attention_mask = torch.cat([instructions['attention_mask'],
            #             torch.zeros(([padding_length] + list(instructions['attention_mask'].shape[1:])),
            #             dtype=instructions['attention_mask'].dtype)],
            #             dim=0)


            timesteps = torch.arange(start=0, end=self.max_length, step=1)

            traj_mask = torch.cat([torch.ones(episode_length, dtype=torch.long),
                                    torch.zeros(padding_length, dtype=torch.long)],
                                    dim=0)

            # lxrt ouput part
            lxrt_feature = self.lxrt_feature[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
            lxrt_feature = lxrt_feature.numpy()
            lxrt_feature = np.concatenate((lxrt_feature,
                                        np.zeros(([padding_length] + list(lxrt_feature.shape[1:])),
                                        dtype= lxrt_feature.dtype)),
                                        axis=0)
        return lxrt_feature, rtg, traj_mask, timesteps
        #return  timesteps, states, actions, rtg, traj_mask, instructions_input_ids, instructions_token_type_ids, instructions_attention_mask, visual_feats, visual_pos


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
