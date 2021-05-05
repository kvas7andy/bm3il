import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), num_outputs=1, activation='tanh', last_sigmoid=True):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
            
        self.last_sigmoid = last_sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, num_outputs)
        #self.logic.weight.data.mul_(0.1)
        #self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        if self.last_sigmoid:
            prob = torch.sigmoid(self.logic(x))
        else:
            prob = self.logic(x)
        return prob
