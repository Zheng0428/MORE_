import torch
import numpy as np
HIDDEN_NUM = 5000
len_i = 36 + 20          #state's patch:36 and text's len:20
len_o = 1                #turn to 1 token

x = torch.rand(32, 32, 100, 2304)
split_size = 768
query = x.split(split_size, dim=2)
a = 1