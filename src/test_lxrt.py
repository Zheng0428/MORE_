import torch
import torch.nn as nn
from torch.utils.data import Dataset
from param import args
import sys
sys.path.append("./src")
from lxrt.entry import LXRTEncoder,convert_sents_to_features
from tasks.more_model import MAX_VQA_LENGTH
from lxrt.tokenization import BertTokenizer
from torch.utils.data.dataloader import DataLoader
ACTION_SPACE_DIC={
    0:"Move to the left direction.",
    1:"Move to the right direction",
    2:"Move upwards",
    3:"Switch the current state or setting of something",
    4:"Grab or take an item",
    5:"Release or let go of an item",
    6:"Indicates the completion of a task or a no-operation action"
}
ACTION_SPACE_LIST=[
    "Move to the left direction.",
    "Move to the right direction",
    "Move upwards",
    "Switch the current state or setting of something",
    "Grab or take an item",
    "Release or let go of an item",
    "Indicates the completion of a task or a no-operation action"
]
class LXMERT(nn.Module):
    def __init__(self, action_space, max_len):
        super().__init__()
        self.action_space = action_space
        self.max_len = max_len
        self.lxmert = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode = 'l'
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
    def forward(self, actions, states, pos):
        action_feature = convert_sents_to_features(
            self.action_space, self.max_len, self.tokenizer)
        x = self.lxmert(actions, (states, pos), action_feature) 
        lxrt_out = torch.cat((x[0], x[1]), 1)
        return lxrt_out
class DataSample(Dataset):
    def __init__(self, action, state, pos):
        self.action = action.view(-1, 1)
        self.state = state.view(-1, 36, 2048)
        self.pos = pos.view(-1, 36, 4)
    def __len__(self):
        return len(self.action)

    def __getitem__(self, index):
        action = self.action[index]
        state = self.state[index]
        pos = self.pos[index]
        return action, state, pos

'''
return:(b, l, 56, 768)
'''
model = LXMERT(ACTION_SPACE_LIST, MAX_VQA_LENGTH)
actions = torch.randint(low=0, high=7, size=(32, 1000))
states = torch.rand(32, 1000, 36, 2048)
poses = torch.rand(32, 1000, 36, 4)


data = DataSample(actions, states, poses)
lxrt_data_loader = DataLoader(             
        data,
        batch_size = 32,
        shuffle = False,
        pin_memory = True,
        drop_last = True
        ) 
a = len(lxrt_data_loader)
for action, state, pos in lxrt_data_loader:
    model, action, state, pos = model.to('cuda'), action.to('cuda'), state.to('cuda'), pos.to('cuda')
    x = model(action, state, pos)
    try:
        lxrt_feature = torch.cat((lxrt_feature, x), dim=0)
    except NameError:
        lxrt_feature = x
a = 1