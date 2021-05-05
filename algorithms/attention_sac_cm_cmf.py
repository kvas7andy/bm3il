import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics_multiType2 import AttentionCritic
from torch.autograd import Variable
from torch import Tensor
import numpy as np

MSELoss = torch.nn.MSELoss()

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 type2agent, agent2type,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 policy_contain_mask = False,
                 meanField = False,
                 mfApproxVersion = 1,
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
            mfApproxVersion = 1: attention over all mf, then concatenate with s + mf of self type
            mfApproxVersion = 2: attention over all mf, then concatenate with s
        """
        self.nagents = len(sa_size)
        self.ntypes = len(type2agent)
        self.type2agent = type2agent
        self.agent2type = agent2type
        self.meanField = meanField
        self.mfApproxVersion = mfApproxVersion

        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        self.critic = AttentionCritic(sa_size, type2agent=self.type2agent, agent2type=self.agent2type, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads, mfApproxVersion=self.mfApproxVersion)
        self.target_critic = AttentionCritic(sa_size, type2agent=self.type2agent, agent2type=self.agent2type, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads, mfApproxVersion=self.mfApproxVersion)
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
        
        assert(self.meanField == True)

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, masks=None, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        if masks is not None:
            return [a.step(obs, mask, explore=explore) for a, obs, mask in zip(self.agents,
                                                                   observations, masks)]
        else:
            return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                   observations)]

    def target_step(self, observations, masks=None, explore=False):
        if masks is not None:
            return [pi(obs, mask) for pi, obs, mask in zip(self.target_policies,
                                                                   observations, masks)]
        else:
            return [pi(obs) for pi, obs in zip(self.target_policies, observations)]
    
    def next_action_mean(self, sample):
        obs, acs, mf_cur, masks, rews, next_obs, dones = sample
        next_acs = []
        for pi, mask, ob in zip(self.target_policies, masks, next_obs):
            curr_next_ac = pi(ob, mask)
            next_acs.append(curr_next_ac)
        if next_acs[0].device.type=='cuda':
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        listMean = lambda x: np.array([sum(y)/len(y) for y in zip(*x)])
        next_acs_mftype = [[listMean(oi) for oi in zip(*[next_acs[ai].data.cpu().numpy() for ai in self.type2agent[ti]])] for ti in range(self.ntypes)]
        next_mfacs = [cast(next_acs_mftype[self.agent2type[ai]]) for ai in range(self.nagents)]
        return next_mfacs
    
    def update_critic(self, sample, mf_next, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        if self.policy_contain_mask:
            obs, acs, mf_cur, masks, rews, next_obs, dones = sample
        else:
            obs, acs, mf_cur, rews, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        if self.policy_contain_mask:
            for pi, mask, ob in zip(self.target_policies, masks, next_obs):
                curr_next_ac, curr_next_log_pi = pi(ob, mask, return_log_pi=True)
                next_acs.append(curr_next_ac)
                next_log_pis.append(curr_next_log_pi)
        else:
            for pi, ob in zip(self.target_policies, next_obs):
                curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
                next_acs.append(curr_next_ac)
                next_log_pis.append(curr_next_log_pi)
        
        if self.meanField:
            #if next_acs[0].device.type=='cuda':
            #    cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
            #else:
            #    cast = lambda x: Variable(Tensor(x), requires_grad=False)
            #listMean = lambda x: np.array([sum(y)/len(y) for y in zip(*x)])
            #next_acs_mftype = [[listMean(oi) for oi in zip(*[next_acs[ai].data.cpu().numpy() for ai in self.type2agent[ti]])] for ti in range(self.ntypes)]
            #next_mfacs = [cast(next_acs_mftype[self.agent2type[ai]]) for ai in range(self.nagents)]
            trgt_critic_in = list(zip(next_obs, mf_next, next_acs))
            critic_in = list(zip(obs, mf_cur, acs))
        else:
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
            obs, acs, mf_cur, masks, rews, next_obs, dones = sample
        else:
            obs, acs, mf_cur, rews, next_obs, dones = sample
            
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        if self.policy_contain_mask:
            for a_i, pi, mask, ob in zip(range(self.nagents), self.policies, masks, obs):
                curr_ac, probs, log_pi, pol_regs, ent = pi(
                    ob, mask, return_all_probs=True, return_log_pi=True,
                    regularize=True, return_entropy=True)
                if logger is not None:
                    logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
                                  self.niter)
                samp_acs.append(curr_ac)
                all_probs.append(probs)
                all_log_pis.append(log_pi)
                all_pol_regs.append(pol_regs)
        else:
            for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
                curr_ac, probs, log_pi, pol_regs, ent = pi(
                    ob, return_all_probs=True, return_log_pi=True,
                    regularize=True, return_entropy=True)
                if logger is not None:
                    logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
                                  self.niter)
                samp_acs.append(curr_ac)
                all_probs.append(probs)
                all_log_pis.append(log_pi)
                all_pol_regs.append(pol_regs)
                
        if self.meanField:
            critic_in = list(zip(obs, mf_cur, samp_acs))
        else:
            critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='cuda'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if (device == 'cuda') or (device == 'gpu'):
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
            a.target_policy.eval()
        if (device == 'cuda') or (device == 'gpu'):
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, mf_len=None, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, policy_contain_mask=False, observation_discrete=False, mfApproxVersion=1,
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
        meanField=True
        if mf_len is None:
            meanField=False
            mf_len = [0 for _ in range(len(env.type2agent))]

        for acsp, obsp, typemfl in zip(env.action_space,
                              env.observation_space, mf_len):
            if observation_discrete:
                agent_init_params.append({'num_in_pol': obsp.n,
                                          'num_out_pol': acsp.n})
                sa_size.append((obsp.n, acsp.n, typemfl))
            else:
                agent_init_params.append({'num_in_pol': obsp.shape[0],
                                          'num_out_pol': acsp.n})
                sa_size.append((obsp.shape[0], acsp.n, typemfl))


        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size,
                     'type2agent': env.type2agent,
                     'agent2type': env.agent2type,
                     'policy_contain_mask': policy_contain_mask,
                     'meanField':meanField,
                     'mfApproxVersion': mfApproxVersion
                    }
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
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance