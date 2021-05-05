import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.svgd import RBF, phi

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, type2agent, agent2type, hidden_dim=32, norm_in=True, attend_heads=1, attend_dim=None):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
            mf_encoders (critic_encoders): input meanfield
            critics: input the attention result
            state_mf_encoders (state_encoders): input observation + meanfield
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.ntypes = len(type2agent)
        self.type2agent = type2agent
        self.agent2type = agent2type
        self.attend_heads = attend_heads

        self.critics = nn.ModuleList()
        self.state_mf_encoders = nn.ModuleList()
        
        if attend_dim is None:
            attend_dim = hidden_dim // attend_heads
        mfdim = sum([sa_sizes[tv[0]][2] for tk, tv in self.type2agent.items()])
        
        # iterate over agents
        for sdim, adim, _ in sa_sizes:
            odim = adim
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim+mfdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim+mfdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_mf_encoders.append(state_encoder)
        
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.fully_shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors]

    def fully_shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.fully_shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.fully_shared_parameters():
            p.grad.data.mul_(1. / self.nagents)
        for ti, module in enumerate(self.mf_encoders):
            for p in module.parameters():
                p.grad.data.mul_(1. / len(self.type2agent[ti]))
            

    def svgd_att_grads(self):
        """
        compute svgd grads for shared parameters
        """
        attHeadsModules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors]
        for idxModule, module in enumerate(attHeadsModules):
            dataDict = [[] for _ in range(self.attend_heads)]
            gradDict = [[] for _ in range(self.attend_heads)]
            for idxHead in range(self.attend_heads):
                for param in module[idxHead].parameters():
                    dataDict[idxHead].append(param.data)
                    gradDict[idxHead].append(param.grad)
            for idx, (w, grad) in enumerate(zip(zip(*dataDict), zip(*gradDict))):
                X = torch.stack(([wc.view(-1) for wc in w]),dim=0)
                score_func = torch.stack(([gc.view(-1) for gc in grad]),dim=0)
                svgdGrad = phi(score_func, RBF, X)
                for idxHead in range(self.attend_heads):
                    gradDict[idxHead][idx] = svgdGrad[idxHead].view(grad[idxHead].shape).data
            for idxHead in range(self.attend_heads):
                for param_cur, param_best in zip(attHeadsModules[idxModule][idxHead].parameters(), gradDict[idxHead]):
                    param_cur.grad = param_best
    
    def forward(self, inps, mfs, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(self.nagents)
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        states_mf = [torch.cat((s, mfs), dim=1) for s, m in inps]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_mf_encoders[a_i](states_mf[a_i]) for a_i in agents]
        
        all_rets = []
        for i, a_i in enumerate(agents):
            #head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
            #                   .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = s_encodings[i]
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            #if logger is not None:
            if 0:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
