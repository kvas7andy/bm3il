import copy
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.latent_ce_dis_rnn_agents import LatentPolicy, DiscreteLatentPolicy

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0, args=None, base_policy=None,
                 embed_net = None, latent_net = None, inference_net = None, dis_net = None, fc2_w_nn = None, fc2_b_nn = None):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """

        #self.base_policy = base_policy
        self.latent = None
        self.latent_infer = None
        self.hidden_state = None

        wr_bck = args.writer
        args.writer = None
        new_args = copy.deepcopy(args)
        args.writer = wr_bck

        self.policy = DiscreteLatentPolicy(num_in_pol, num_out_pol,
                                   new_args, embed_net, latent_net, inference_net, dis_net,
                                   base_policy, fc2_w_nn, fc2_b_nn)
        self.target_policy = copy.deepcopy(self.policy)
        # LatentPolicy(num_in_pol,
        #                                   num_out_pol,
        #                                   args,
        #                                   hidden_dim=hidden_dim,
        #                                   onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, hidden_state, t, mask=None, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        agent_out = self.policy(obs, hidden_state, t, mask, sample=explore) # obs == [tensor(st).to(dtype).unsqueeze(0) for st in state][i] ==
                # obs ~{diverse_spread.py::def observation():}~ size(obs) = 18: [vel_x, vel_y, pos_x, pos_y, rel_landmark_1_x, rel_landmark_1_y,
        #   rel_landmark_2_x, rel_landmark_2_y,rel_landmark_3_x, rel_landmark_3_y,
        #   pos_other_agent_1_x, pos_other_agent_1_y, pos_other_agent_2_x, pos_other_agent_1_x,
        #   comm_other_agent_1_x, comm_other_agent_1_y, comm_other_agent_2_x, comm_other_agent_1_x]

    #  diverse_spread: np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        self.latent = self.policy.latent
        self.latent_infer = self.policy.latent_infer
        self.hidden_state = self.policy.hidden_state
        return agent_out
    
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
