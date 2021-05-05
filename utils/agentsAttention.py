from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.attention_policy import AttentionPolicy

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, sa_size, hidden_dim=64, attend_heads=2,
                 lr=1e-3, weight_decay=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = AttentionPolicy(sa_size, hidden_dim=hidden_dim,
                                     attend_heads=attend_heads)
        self.target_policy = AttentionPolicy(sa_size, hidden_dim=hidden_dim,
                                     attend_heads=attend_heads)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)

    def step(self, obs, mask=None, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, mask=mask, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
