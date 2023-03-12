import torch.nn as nn
import torch 

n_embd = 768
rtgs = torch.randint(low=0, high=100, size=(32, 100)).unsqueeze(2)
ret_emb = nn.Sequential(nn.Linear(1, n_embd), nn.Tanh())
rtg_embeddings = ret_emb(rtgs.type(torch.float32))
point = 1