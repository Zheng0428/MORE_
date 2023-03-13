import torch
HIDDEN_NUM = 5000
len_i = 36 + 20          #state's patch:36 and text's len:20
len_o = 1                #turn to 1 token
lxrt_feature = torch.rand(32000,56,768)
lxrt_feature = lxrt_feature.view(32, 1000, lxrt_feature.shape[1], lxrt_feature.shape[2])
tmp = 1