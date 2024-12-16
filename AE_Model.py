# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:57:47 2024

@author: MA11201
"""
import torch
import torch.nn as nn

def LinearBnTanh(input_dim, out_dim):
    linear_bn_tanh = nn.Sequential(
        nn.Linear(input_dim, out_dim), nn.Tanh())
    return linear_bn_tanh

class AE(nn.Module):
    ''' Vanilla AE '''
    def __init__(self, input_dim):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            LinearBnTanh(input_dim, 8),
            LinearBnTanh(8, 4),
            LinearBnTanh(4, 2),
            LinearBnTanh(2, 1))




        self.decoder = nn.Sequential(
            LinearBnTanh(1, 2),
            LinearBnTanh(2, 4),
            LinearBnTanh(4, 8),
            nn.Linear(8, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output




