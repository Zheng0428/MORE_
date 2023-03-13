import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sys
from tqdm import tqdm
sys.path.append("./src")
from param import args
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
    def __init__(self, action_space, max_len, device):
        super().__init__()
        self.action_space = action_space
        self.max_len = max_len
        self.device = device
        self.lxmert = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            device = device,
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
        self.action = action
        self.state = state
        self.pos = pos

    def __len__(self):
        return len(self.action)

    def __getitem__(self, index):
        action = self.action[index]
        state = self.state[index]
        pos = self.pos[index]
        return action, state, pos
    
class LXMTDataLoad(nn.Module):
    def __init__(self, bs, device):
        super().__init__()
        self.bs = bs
        self.device = device
        self.model = LXMERT(ACTION_SPACE_LIST, MAX_VQA_LENGTH, device)
        # Load lxmert_Encoder weights
        self.model.lxmert.load(args.load_lxmert)

    def forward(self, actions, states, poses):
        data = DataSample(actions, states, poses)
        lxrt_data_loader = DataLoader(             
                data,
                batch_size = self.bs,
                shuffle = False,
                pin_memory = True,
                drop_last = False
                ) 
        a = len(lxrt_data_loader)
        iter_wrapper = (lambda x: tqdm(x, total=len(lxrt_data_loader))) if True else (lambda x: x)
        model = self.model.to(self.device)
        model.eval()
        for i, (action, state, pos) in iter_wrapper(enumerate(lxrt_data_loader)):
            with torch.no_grad():
                action, state, pos = action.to(self.device), state.to(self.device), pos.to(self.device)
                x = model(action, state, pos)
            try:
                lxrt_feature = torch.cat((lxrt_feature, x), dim=0)
            except NameError:
                lxrt_feature = x
        torch.cuda.empty_cache()
        return lxrt_feature
    
'''
return:(b, l, 56, 768)
'''
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# actions = torch.randint(low=0, high=7, size=(32, 1000))
# states = torch.rand(32, 1000, 36, 2048)
# poses = torch.rand(32, 1000, 36, 4)
# lxrt_dataload_model = LXMTDataLoad(128, device)
# lxrt_feature = lxrt_dataload_model(actions, states, poses)
# a = 1

