import torch.nn as nn
import torch
from models.spectral_normalization import SpectralNorm

class VAEDiscriminator(nn.Module):
    def __init__(self, num_inputs, num_outputs, sigmoid_out = True, sn = False, w_init = False, test=True, hidden_size_enc=(64,), hidden_size_dec=(64,), encode_size = 64, activation='tanh',dropout=False,slope=0.1,dprob=0.2):
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
        self.sigmoid_out = sigmoid_out
        """
        class Q(nn.Module):
            def __init__(self):
                super(Q,self).__init__()
            def forward(self,x):
        """     
        self.encoder = nn.Sequential()
        last_dim = num_inputs
        for ih,nh in enumerate(hidden_size_enc):
            if sn:
                self.encoder.add_module('enc_lin'+str(ih),SpectralNorm(nn.Linear(last_dim, nh),w_init))
            else:
                self.encoder.add_module('enc_lin'+str(ih),nn.Linear(last_dim, nh))
            self.encoder.add_module('enc_act'+str(ih),self.activation)
            if dropout:
                self.encoder.add_module('enc_dro'+str(ih),nn.Dropout(p=dprob))
            last_dim = nh
        if sn:
            self.encoder.add_module('encoder_out',SpectralNorm(nn.Linear(last_dim, encode_size*2),w_init))
        else:
            self.encoder.add_module('encoder_out',nn.Linear(last_dim, encode_size*2))
        
        self.decoder = nn.Sequential()
        last_dim = encode_size
        
        # to be deleted
        if test:
            self.decoder.add_module('dec_act',self.activation)
            if dropout:
                self.decoder.add_module('dec_dro',nn.Dropout(p=dprob))
        
        for ih,nh in enumerate(hidden_size_dec):
            if sn:
                self.decoder.add_module('dec_lin'+str(ih),SpectralNorm(nn.Linear(last_dim, nh),w_init))
            else:
                self.decoder.add_module('dec_lin'+str(ih),nn.Linear(last_dim, nh))
            self.decoder.add_module('dec_act'+str(ih),self.activation)
            if dropout:
                self.decoder.add_module('dec_dro'+str(ih),nn.Dropout(p=dprob))
            last_dim = nh
        self.logic = nn.Linear(last_dim, num_outputs)
        if sn:
            self.decoder.add_module('decoder_out',SpectralNorm(self.logic,w_init))
        else:
            self.decoder.add_module('decoder_out',self.logic)

    def forward(self, x, mean_mode=True, v_scale = 1):
        """
        forward pass of the module
        :param x: input image tensor [Batch_size x 3 x height x width]
        :param mean_mode: decides whether to sample points or use means directly
        :return: prediction scores (Linear), mus and sigmas: [Batch_size x 1]
        """

        mid = self.encoder(x)
        halfpoint = mid.shape[-1] // 2
        mus, sigmas = mid[:, :halfpoint], mid[:, halfpoint:]
        sigmas = torch.sigmoid(sigmas)  # sigmas are restricted to be from 0 to 1

        # difference between generator training and discriminator
        # training (please refer the paper for more info.)
        if not mean_mode:
            # sample points from this gaussian distribution
            # this is for the discriminator
            out = (torch.randn_like(mus).to(x.device) * sigmas * v_scale) + mus
        else:
            # just use the means forward
            # this is for generator
            out = mus

        # apply the final fully connected layer
        if self.sigmoid_out:
            out = torch.sigmoid(self.decoder(out))
        else:
            out = self.decoder(out)

        # return the predictions, mus and sigmas
        return out, mus, sigmas
