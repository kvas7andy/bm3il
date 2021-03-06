import torch.nn as nn
import torch
from models.spectral_normalization import SpectralNorm


class SNDiscriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(SpectralNorm(nn.Linear(last_dim, nh)))
            last_dim = nh
        
        self.logic_tmp = nn.Linear(last_dim, 1)
        #self.logic_tmp.weight.data.mul_(0.1)
        #self.logic_tmp.bias.data.mul_(0.0)
        self.logic = SpectralNorm(self.logic_tmp)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        prob = self.logic(x)
        return prob
