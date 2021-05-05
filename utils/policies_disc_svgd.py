import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample
from itertools import chain
from utils.svgd import RBF, phi

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, pol_net_nums=4,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()
        
        self.pol_net_nums = pol_net_nums
        self.policies = nn.ModuleList()
        
        for pi in range(self.pol_net_nums):
            policy = nn.Sequential()
            if norm_in:
                policy.add_module('in_fn', nn.BatchNorm1d(input_dim,
                                                            affine=False))
            policy.add_module('fc1', nn.Linear(input_dim, hidden_dim))
            policy.add_module('fc2', nn.Linear(hidden_dim, hidden_dim))
            policy.add_module('fc3', nn.Linear(hidden_dim, out_dim))
            policy.add_module('nonlin', nn.LeakyReLU())
            self.policies.append(policy)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        outList = [self.policies[pi](X) for pi in range(self.pol_net_nums)]
        out = torch.mean(torch.stack(outList, dim=0), dim=0)
        return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, mask=None, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        outfull = super(DiscretePolicy, self).forward(obs)
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
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
    
    def scale_policy_grads(self):
        """
        Scale gradients for critic since they are seperated into multiple pieces
        """
        for p in chain(*[self.policies[pi].parameters() for pi in range(self.pol_net_nums)]):
            p.grad.data.mul_(1. * self.pol_net_nums)
    
    def svgd_policy_grads(self):
        """
        compute svgd grads for shared parameters
        """
        dataDict = [[] for _ in range(self.pol_net_nums)]
        gradDict = [[] for _ in range(self.pol_net_nums)]
        for pi in range(self.pol_net_nums):
            for param in self.policies[pi].parameters():
                dataDict[pi].append(param.data)
                gradDict[pi].append(param.grad)
        for idx, (w, grad) in enumerate(zip(zip(*dataDict), zip(*gradDict))):
            X = torch.stack(([wc.view(-1) for wc in w]),dim=0)
            score_func = torch.stack(([gc.view(-1) for gc in grad]),dim=0)
            svgdGrad = phi(score_func, RBF, X)
            for pi in range(self.pol_net_nums):
                gradDict[pi][idx] = svgdGrad[pi].view(grad[pi].shape).data
        for pi in range(self.pol_net_nums):
            for param_cur, param_best in zip(self.policies[pi].parameters(), gradDict[pi]):
                param_cur.grad = param_best
