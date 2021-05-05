import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agentsAttention import AttentionAgent
from utils.critics import AttentionCritic

MSELoss = torch.nn.MSELoss()

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4, pol_attend_heads=4,
                 policy_contain_mask = False,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)

        #self.agents = [AttentionAgent(lr=pi_lr,
        #                              hidden_dim=pol_hidden_dim,
        #                              **params)
        #                 for params in agent_init_params]
        self.agents = AttentionAgent(sa_size, hidden_dim=pol_hidden_dim,
                                      attend_heads=pol_attend_heads, lr=pi_lr)
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        self.policy_contain_mask = policy_contain_mask

    @property
    def policies(self):
        return self.agents.policy

    @property
    def target_policies(self):
        return self.agents.target_policy

    def step(self, observations, masks=None, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return self.agents.step(observations, mask=masks, explore=explore)

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        if self.policy_contain_mask:
            obs, acs, masks, rews, next_obs, dones = sample
        else:
            obs, acs, rews, next_obs, dones = sample
            masks = None
        # Q loss
        next_acs, next_log_pis = zip(*self.target_policies(next_obs, mask=masks, return_log_pi=True))
            
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        if self.policy_contain_mask:
            obs, acs, masks, rews, next_obs, dones = sample
        else:
            obs, acs, rews, next_obs, dones = sample
            masks = None
        
        samp_acs, all_probs, all_log_pis, all_pol_regs, ent, all_pol_attend_regs = zip(*self.policies(
                    obs, mask=masks, return_all_probs=True, return_log_pi=True,
                    regularize=True, return_entropy=True, regularize_attend=True,))

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        
        all_pol_loss = 0
        for probs, log_pi, pol_regs, pol_attend_regs, (q, all_q) in zip(all_probs, all_log_pis, 
                                                       all_pol_regs, all_pol_attend_regs, critic_rets):
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            for reg in pol_attend_regs:
                pol_loss += 1e-3 * reg  # regularizing attention
            all_pol_loss += pol_loss
                
        # don't want critic to accumulate gradients from policy loss
        disable_gradients(self.critic)
        all_pol_loss.backward()
        enable_gradients(self.critic)
        self.agents.policy.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agents.policy.parameters(), 0.5)
        self.agents.policy_optimizer.step()
        self.agents.policy_optimizer.zero_grad()


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.agents.target_policy, self.agents.policy, self.tau)

    def prep_training(self, device='cuda'):
        self.critic.train()
        self.target_critic.train()
        self.agents.policy.train()
        self.agents.target_policy.train()
        if (device == 'cuda') or (device == 'gpu'):
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.agents.policy = fn(self.agents.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.agents.target_policy = fn(self.agents.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        self.agents.policy.eval()
        if (device == 'cuda') or (device == 'gpu'):
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.agents.policy = fn(self.agents.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': {'policy': self.agents.policy.state_dict(),
                                       'target_policy': self.agents.target_policy.state_dict(),
                                       'policy_optimizer': self.agents.policy_optimizer.state_dict()},
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, pol_attend_heads=4, policy_contain_mask=False, observation_discrete=False,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            if observation_discrete:
                agent_init_params.append({'num_in_pol': obsp.n,
                                          'num_out_pol': acsp.n})
                sa_size.append((obsp.n, acsp.n))
            else:
                agent_init_params.append({'num_in_pol': obsp.shape[0],
                                          'num_out_pol': acsp.n})
                sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'pol_attend_heads': pol_attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size,
                     'policy_contain_mask': policy_contain_mask}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        agent_params = save_dict['agent_params']
        instance.agents.policy.load_state_dict(agent_params['policy'])
        instance.agents.target_policy.load_state_dict(agent_params['target_policy'])
        instance.agents.policy_optimizer.load_state_dict(agent_params['policy_optimizer'])
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance