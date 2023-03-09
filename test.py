import torch
import torch.nn as nn

input = torch.randn(64, 10, 128)  # 64个样本，每个样本10个特征，每个特征有128个值
layer_norm = nn.LayerNorm(128)  # 对每个样本的所有特征进行归一化
output = layer_norm(input)  # 归一化后的输出张量的形状与输入相同

print(output.shape)
