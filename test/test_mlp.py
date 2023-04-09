import torch
import torch.nn as nn


class MLPModel(torch.nn.Module):
    def __init__(self, input_len, hidden_size, output_len):
        super(MLPModel,self).__init__()
        self.output_len = output_len
        self.fc1 = nn.Linear(input_len * 768, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_len * 768)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 2, 3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        output = x.view(x.size(0), x.size(1), 768)
        return output
    
model = MLPModel(56,5000,1)
a = torch.rand(32, 1000, 56, 768)
output = model(a)
c = 1
