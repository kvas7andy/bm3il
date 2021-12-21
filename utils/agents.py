from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies_disc import DiscretePolicy

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, mask=None, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, mask, sample=explore)
    
    def scale_shared_grads(self,downScale):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.policy.parameters():
            p.grad.data.mul_(1. / downScale)
                
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params, device='cpu'):
        self.policy.load_state_dict(params['policy'])
        self.policy.to(device)
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_policy.to(device)
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
