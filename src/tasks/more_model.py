# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import transformers
from param import args
from lxrt.entry import LXRTEncoder,convert_sents_to_features
from lxrt.modeling import BertLayerNorm, GeLU, MLPModel
from tasks.gpt2 import GPT2LMHeadModel
import numpy as np

########################################
import torch
from torch.nn.parameter import Parameter
import math, time
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from lxrt.tokenization import BertTokenizer
########################################
# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20
HIDDEN_NUM = 5000
HIDDEN_LAYERS = 6
pad_id = 0
class MOREModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2") #load tokenizer
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        transformers.logging.set_verbosity_error()
        # Fix the last four layers of the model and train only the first two layers
        self.more_decoder = GPT2LMHeadModel.from_pretrained('gpt2', num_hidden_layers = HIDDEN_LAYERS)
        for i, layer in enumerate(self.more_decoder.transformer.h):
            if i >= HIDDEN_LAYERS - 4:
                for param in layer.parameters():
                    param.requires_grad = False
        #build compresstion model
        len_i = 36 + 20          #state's patch:36 and text's len:20
        len_o = 1                #turn to 1 token
        self.compression_model = MLPModel(len_i, HIDDEN_NUM, len_o)

    def forward(self, lxmert_out, rtg, traj_mask, timesteps):
        """
        b -- batch_size, l -- traj_len, d -- dim 

        :param lxmert: (b, l, 56, d)
        :param rtg:  (b, l)
        :param actions: (b, l) Type -- list of string
        :param traj_mask: (b, l)
        :param timesteps: (b, l)
        :return: (b, l, 50257) 
        """
        lxmert_out = self.compression_model(lxmert_out)
        output = self.more_decoder(inputs_embeds = lxmert_out, labels = lxmert_out, attention_mask = traj_mask, position_ids = timesteps, rtg = rtg)        #Be sure to pay attention to whether the input sequences are of the same length  #past_key_values = past 后面有时间可以加上
        return output
