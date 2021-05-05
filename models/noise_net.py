import torch.nn as nn
import torch
from models.spectral_normalization import SpectralNorm

class NoiseNet(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), encode_size = 64, activation='tanh',dropout=False,slope=0.1,dprob=0.2,sn=True):
        super().__init__()
        if activation == 'tanh':
            self.activation = nn.Tanh() #torch.tanh
        elif activation == 'relu':
            self.activation = nn.ReLU() #torch.relu: function, nn.ReLU: layer
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid() #torch.sigmoid
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(slope)
        self.encode_size = encode_size
        self.encoder = nn.Sequential()
        last_dim = num_inputs
        for ih,nh in enumerate(hidden_size):
            if sn:
                self.encoder.add_module('enc_lin'+str(ih),SpectralNorm(nn.Linear(last_dim, nh)))
            else:
                self.encoder.add_module('enc_lin'+str(ih),nn.Linear(last_dim, nh))
            self.encoder.add_module('enc_act'+str(ih),self.activation)
            if dropout:
                self.encoder.add_module('enc_dro'+str(ih),nn.Dropout(p=dprob))
            last_dim = nh
        if sn:
            self.encoder.add_module('encoder_out',SpectralNorm(nn.Linear(last_dim, encode_size)))
        else:
            self.encoder.add_module('encoder_out',nn.Linear(last_dim, encode_size))

    def forward(self, x):
        out = self.encoder(x)
        return out
