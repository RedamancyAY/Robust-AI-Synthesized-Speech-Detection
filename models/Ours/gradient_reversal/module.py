from .functional import revgrad
import torch
from torch import nn

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.step = 0
        self.total_step = 200000
    
    def forward(self, x):
        self.step = self.step + 1
        # alpha = 2/ (1 + torch.exp(-10 * self.alpha * (self.step/self.total_step) )) - 1
        # print(alpha)
        return revgrad(x, self.alpha)