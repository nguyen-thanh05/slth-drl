import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from GetSubnet import GetSubnet


class SupermaskConv2d(nn.Conv2d):
    def __init__(self, sparsity=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        self.weight.requires_grad = False
        
    def forward(self, input):
        subnet_mask = GetSubnet.apply(self.scores, torch.zeros(self.scores.size()), torch.ones(self.scores.size()), self.sparsity)
        w = self.weight * subnet_mask
        x = F.conv2d(input, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
    
    def get_scores(self):
        return self.scores
    

class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(7))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        self.weight.requires_grad = False
        
    def forward(self, input):
        subnet_mask = GetSubnet.apply(self.scores, torch.zeros(self.scores.size()), torch.ones(self.scores.size()), self.sparsity)
        w = self.weight * subnet_mask
        x = F.linear(input, w, self.bias)
        return x

    def get_scores(self):
        return self.scores
