import torch.nn as nn
import torch

class AEDiscriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), encode_size = 64, activation='tanh',dropout=False,slope=0.1,dprob=0.2):
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
            self.encoder.add_module('enc_lin'+str(ih),nn.Linear(last_dim, nh))
            self.encoder.add_module('enc_act'+str(ih),self.activation)
            if dropout:
                self.encoder.add_module('enc_dro'+str(ih),nn.Dropout(p=dprob))
            last_dim = nh
        self.encoder.add_module('encoder_out',nn.Linear(last_dim, encode_size))
        #self.dec_layers = nn.Modulelist()
        #last_dim = encode_size
        #for nh in hidden_size:
        #    self.dec_layers.append(nn.Linear(last_dim, nh))
        #    self.dec_layers.append(self.activation)
        #    if dropout:
        #        self.dec_layers.append(nn.Dropout(p=dprob))
        #    last_dim = nh
        #self.decoder = nn.Sequential(self.dec_layers)
        self.decoder = nn.Sequential()
        last_dim = encode_size
        for ih,nh in enumerate(hidden_size):
            self.decoder.add_module('dec_lin'+str(ih),nn.Linear(last_dim, nh))
            self.decoder.add_module('dec_act'+str(ih),self.activation)
            if dropout:
                self.decoder.add_module('dec_dro'+str(ih),nn.Dropout(p=dprob))
            last_dim = nh
        self.logic = nn.Linear(last_dim, num_inputs)
        #self.logic.weight.data.mul_(0.1)
        #self.logic.bias.data.mul_(0.0)
        self.decoder.add_module('decoder_out',self.logic)

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return code,out

