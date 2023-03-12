from lxrt.modeling import BertLayerNorm, GeLU, MLPModel
import torch
HIDDEN_NUM = 5000
len_i = 36 + 20          #state's patch:36 and text's len:20
len_o = 1                #turn to 1 token
compression_model = MLPModel(len_i, HIDDEN_NUM, len_o)
lxmert_out = torch.rand(32,1000,56,768)
lxmert_out = compression_model(lxmert_out)
tmp = 1