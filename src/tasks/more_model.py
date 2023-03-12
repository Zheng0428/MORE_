# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import transformers
from param import args
from lxrt.entry import LXRTEncoder,convert_sents_to_features
from lxrt.modeling import BertLayerNorm, GeLU, MLPModel
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
pad_id = 0
class MOREModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Build LXRT encoder
        self.more_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode = 'l'
        )
        self.lxrt_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2") #load tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        transformers.logging.set_verbosity_error()
        self.more_decoder = transformers.GPT2LMHeadModel.from_pretrained('gpt2', num_hidden_layers = 6)
        #build compresstion model
        len_i = 36 + 20          #state's patch:36 and text's len:20
        len_o = 1                #turn to 1 token
        self.compression_model = MLPModel(len_i, HIDDEN_NUM, len_o)

    def forward(self, lxmert_out, rtg, actions, traj_mask, timesteps):
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
        output = self.more_decoder(inputs_embeds = lxmert_out, attention_mask = traj_mask, postion_ids = timesteps)        #Be sure to pay attention to whether the input sequences are of the same length  #past_key_values = past 后面有时间可以加上
        return output
