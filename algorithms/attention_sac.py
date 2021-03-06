import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
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
                 critic_hidden_dim=128, attend_heads=4,
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

        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
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
        if 'custom_policies' in kwargs.keys():
            self.custom_policies = kwargs['custom_policies']
        else:
            raise ValueError("Either init not from env OR no error with 'custom_policies'")

        if 'labeling' in kwargs.keys():
            self.labeling = kwargs['labeling']
        else:
            raise ValueError("Either init not from env OR no error with 'labeling'")

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
        if self.custom_policies is not None:
            observations = [obs.data.numpy()[0] for obs in observations]
            return [torch.tensor(action_ai).unsqueeze(0) for action_ai in self.custom_policies(observations)]
        if self.labeling is not None:
            label_size = 1 #
            if self.labeling is True:
                label_size = 1 #
                label, state = [ag_st[-label_size:] for ag_st in state], [ag_st[:-label_size] for ag_st in state]
        if masks is not None:
            output = [a.step(obs, mask, explore=explore) for a, obs, mask in zip(self.agents,
                                                                   observations, masks)]
        else:
            output = [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                   observations)]
        return output

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        if self.policy_contain_mask:
            obs, acs, masks, rews, next_obs, dones = sample# Transition(state=(s_n1, s_n2,..., s_n{batch_size},
                                                                #action =  (a_n1, a_n2,..., a_n{batch_size}), ... )
        else:
            obs, acs, rews, next_obs, dones = sample
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
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        reguls = 0
        q_loss = 0
        soft_qs = 0
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
                reguls += reg
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
            logger.add_scalar('losses/q_reguls', reguls, self.niter)
        self.niter += 1

    def value_critic(self, sample, soft=True, logger=None, episode_length=50, **kwargs):
        """
        Update central critic for all agents
        """
        if self.policy_contain_mask:
            obs, acs, masks, rews, next_obs, dones = sample# Transition(state=(s_n1, s_n2,..., s_n{batch_size},
                                                                #action =  (a_n1, a_n2,..., a_n{batch_size}), ... )
        else:
            obs, acs, rews, next_obs, dones = sample
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
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, logger=logger, niter=self.niter)
        targets = []
        for a_i, nq, log_pi, pq in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            targets.append(target_q.cpu().numpy())

            if logger is not None:
                logger.add_scalar('agent%i/targets_yi' % a_i, pq.mean(),self.niter)
                logger.add_scalar('agent%i/q_values_mean' % a_i, target_q.mean(), self.niter)
                logger.add_scalar('agent%i/soft_q' % a_i, (-log_pi / self.reward_scale).mean(), self.niter )
        self.niter += 1
        return [rew.cpu().numpy() for rew in rews], [cr.cpu().numpy() for cr in critic_rets], targets
            # not cr[0] because we have no regularizer as output

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        if self.policy_contain_mask:
            obs, acs, masks, rews, next_obs, dones = sample
        else:
            obs, acs, rews, next_obs, dones = sample
            
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
        self.critic.eval()
        self.target_critic.eval()
        for a in self.agents:
            a.policy.eval()
        if (device == 'cuda') or (device == 'gpu'):
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

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
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, policy_contain_mask=False, observation_discrete=False,
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
        if mf_len is not None:
            for acsp, obsp, typemfl in zip(env.action_space,
                                  env.observation_space, mf_len):
                if observation_discrete:
                    agent_init_params.append({'num_in_pol': obsp.n + typemfl,
                                              'num_out_pol': acsp.n})
                    sa_size.append((obsp.n + typemfl, acsp.n))
                else:
                    agent_init_params.append({'num_in_pol': obsp.shape[0] + typemfl,
                                              'num_out_pol': acsp.n})
                    sa_size.append((obsp.shape[0] + typemfl, acsp.n))
        else:
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
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size,
                     'policy_contain_mask': policy_contain_mask}
        init_dict.update({'custom_policies': env.custom_policies})
        init_dict.update({'labeling': env.labeling})
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False, device='cpu'):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=device)
        save_dict['init_dict'].update({'custom_policies': None})
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params, device=device)
        instance.pol_dev = device.type
        instance.trgt_pol_dev = device.type

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.critic.to(device)
            instance.critic_dev = device.type
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.target_critic.to(device)
            instance.trgt_critic_dev = device.type
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance