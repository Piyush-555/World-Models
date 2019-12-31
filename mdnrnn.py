import torch
import torch.nn as nn
from mdn import MDN
# Implementation of Mixture Density Network Layer in PyTorch
# Courtesy of https://github.com/sagelywizard/pytorch-mdn

class MDNRNN(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(32+3, 256) # observation + action

        self.mdn = MDN(256, 32, 5)
    
    def forward(self, z_a, h, c):
        y, (h_next, c_next) = self.lstm(z_a, (h, c))
        pi, sigma, mu = self.mdn(y)
        return pi, sigma, mu, h_next, c_next
    
    @classmethod
    def _init_hidden(self, batch_size):
        return (
            torch.zeros(batch_size, 256),
            torch.zeros(batch_size, 256)
        )
