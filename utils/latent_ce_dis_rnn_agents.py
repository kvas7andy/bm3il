import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
import math
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import time
from utils.misc import onehot_from_logits, categorical_sample


class LatentPolicy(nn.Module):
    def __init__(self, num_in_pol, num_out_pol,
                 args, embed_net, latent_net, inference_net, dis_net, base_policy, fc2_w_nn, fc2_b_nn):
        super(LatentPolicy, self).__init__()
        args = args
        self.args = args
        self.num_in_pol = num_in_pol
        self.num_out_pol = num_out_pol
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0


        self.embed_net = embed_net
        self.latent_net = latent_net
        self.inference_net = inference_net
        self.dis_net = dis_net
        self.base_policy = base_policy

        self.fc2_w_nn = fc2_w_nn
        self.fc2_b_nn = fc2_b_nn

        self.latent = None
        self.hidden_state = None
        self.embed_fc_input_size = num_in_pol ### TODO: if latent output is not just o_i

        # # Base policy
        # self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # # additional parameters

    def forward(self, inputs, hidden_state, t=0):
        inputs = inputs.reshape(-1, self.num_in_pol)
        self.bs = inputs.shape[0]
        h_in = hidden_state.reshape(-1, self.hidden_dim) ## \tau_i^{t-1}

        embed_fc_input = inputs[:, - self.embed_fc_input_size:]  # own features(unit_type_bits+shield_bits_ally)+id

        self.latent = self.embed_net(embed_fc_input) # role-encoder output == (mu & sigma): p(\rho_i^t |o_i^t) - N(mu, sigma)
        self.latent[:, -self.latent_dim:] = th.clamp(th.exp(self.latent[:, -self.latent_dim:]), min=self.args.var_floor)  # Sigma == var
        #self.latent[:, -self.latent_dim:] = th.full_like(self.latent[:, -self.latent_dim:],1.0)

        latent_embed = self.latent.reshape(self.bs, self.latent_dim * 2) # (bs * XnX, latent_dim *2)

        #latent = latent_embed[:, :self.latent_dim]
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2)) # (bs * n, latent_dim *2)
        self.latent_sampled = gaussian_embed.rsample() # (bs * XnX, latent_dim *2)

        self.latent_infer = self.inference_net(th.cat([h_in.detach(), inputs], dim=1))  # q_\xi(\rho_i^t|\tau_i^{t-1}, o_i^t)
        self.latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.latent_infer[:, -self.latent_dim:]),
                                                            min=self.args.var_floor)  # n,mu+var for q_xi
        # self.latent_infer[:, -self.latent_dim:] = th.full_like(self.latent_infer[:, -self.latent_dim:],1.0)
        # gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim],
        #                           (self.latent_infer[:, self.latent_dim:]) ** (1 / 2))
        # latent_infer = gaussian_infer.rsample()
        # Role -> FC2 Params
        latent_params = self.latent_net(self.latent_sampled)# + latent_infer) # (bs *XnX, NN_HIDDEN_SIZE)

        # role decoder
        fc2_w = self.fc2_w_nn(latent_params) # (bs * XnX, args.rnn_hidden_dim * args.n_actions)
        fc2_b = self.fc2_b_nn(latent_params) # (bs * XnX, args.n_actions)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

        #x = F.relu(self.fc1(inputs))  # (bs*n,(obs+act+id)) at time t
        #h = self.rnn(x, h_in)
        h = self.base_policy(inputs, h_in)
        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        actions_out = th.bmm(h, fc2_w) + fc2_b

        h = h.reshape(-1, self.args.rnn_hidden_dim)

        return actions_out.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim)


class DiscreteLatentPolicy(LatentPolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscreteLatentPolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, hidden_state, t, mask=None, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        outfull, hi = super(DiscreteLatentPolicy, self).forward(obs, hidden_state, t)
        self.hidden_state = hi
        if mask is not None:
            out = outfull*mask
        else:
            out = outfull
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi and sample:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
