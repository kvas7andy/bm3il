import torch
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
import torch.distributions as D
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agentsLatent import AttentionAgent
from utils.critics import AttentionCritic
from utils.policies_disc_rnn import BasePolicyNoFC3

MSELoss = torch.nn.MSELoss()

class AttentionSACLatent(object):
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
                 args=None,
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
        self.n_agents = args.n_agents
        self.args = args
        self.i_iter = None

        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim

        # OUR case:
        # agent_init_params.append({'num_in_pol': obsp.n,
        #                           'num_out_pol': acsp.n})
        # sa_size.append((obsp.n, acsp.n))

        st_nip, st_nop = sa_size[0] # (int, int) ==  sa_size.append((obsp.n, acsp.n)) OR obsp.shape[0] {{{env.actions_space, env_observation_space}}} for agents
        # input shape = #  act+obs+id; not (bs*n, act+obs+id)
        for nip, nop in sa_size[1:]:
            if st_nip != nip or st_nop != nop:
                print("Not same input for all agents with shared policies!")
                return

        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        activation_func = nn.LeakyReLU()

        self.hidden_states = None
        self.trgt_hidden_states = None
        self.base = BasePolicyNoFC3(st_nip, st_nop,
                                 hidden_dim=args.rnn_hidden_dim, nonlin=activation_func) # pol_hidden_dim = 128 (fc1, fc2 -> pol_hidden_dim)
        self.trgt_base = BasePolicyNoFC3(st_nip, st_nop,
                                    hidden_dim=args.rnn_hidden_dim, nonlin=activation_func) # pol_hidden_dim <-> args.rnn_hidden_dim

        self.embed_fc_input_size = st_nip# input_shape
        self.embed_net = nn.Sequential(nn.Linear(self.embed_fc_input_size, NN_HIDDEN_SIZE),
                                       # role encoder - p_theta_rho(\rho_i^t|o_i^t)
                                       nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       activation_func,
                                       nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))  # mu+var

        self.inference_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + st_nip, NN_HIDDEN_SIZE),
                                           # trajectory encoder -  q(\rho_i^t|\tau_i^{t-1}, o_it)
                                           nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                           activation_func,
                                           nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))  # mu+var

        self.latent = th.rand(args.n_agents, 1, args.latent_dim * 2)  # (n,mu+var)
        self.latent_infer = th.rand(args.n_agents, 1, args.latent_dim * 2)  # (n,mu+var)

        self.latent_net = nn.Sequential(nn.Linear(args.latent_dim, NN_HIDDEN_SIZE),
                                        # role decoder - g_\theta_h(\rho_i^t)
                                        nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func)

        self.fc2_w_nn = nn.Linear(NN_HIDDEN_SIZE, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_nn = nn.Linear(NN_HIDDEN_SIZE, args.n_actions)

        # DisNet
        self.dis_net = nn.Sequential(nn.Linear(args.latent_dim * 2, NN_HIDDEN_SIZE),
                                     nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                     activation_func,
                                     nn.Linear(NN_HIDDEN_SIZE, 1))
        self.mi = th.rand(args.n_agents * args.n_agents)
        self.dissimilarity = th.rand(args.n_agents * args.n_agents)

        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      args=args,
                                      base_policy = self.base,
                                      embed_net = self.embed_net,
                                      latent_net = self.latent_net,
                                      inference_net = self.inference_net,
                                      dis_net = self.dis_net,
                                      fc2_w_nn = self.fc2_w_nn,
                                      fc2_b_nn = self.fc2_b_nn,
                                      **params)
                         for params in agent_init_params]


        if args.dis_sigmoid:
            print('>>> sigmoid')
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_sigmoid
        else:
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_step

        ### Critic
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
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, t, masks=None, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        if masks is not None:
            agents_out = [a.step(obs, hidden_state, t, mask, explore=explore) for a, obs, hidden_state, mask in zip(self.agents,
                                                                   observations, self.hidden_states, masks)] # [tensor(st).to(dtype).unsqueeze(0) for st in state]
            #self.hidden_states = th.cat([h_in.unsqueeze(0) for h_in, ag_out in self.agents])
            # return [ag_out for h_in, ag_out in h_in_agents_out]

        else:
            agents_out =  [a.step(obs, hidden_state, t, explore=explore) for a, obs, hidden_state in zip(self.agents,
                                                                   observations, self.hidden_states)] # [tensor(st).to(dtype).unsqueeze(0) for st in state]
            #self.hidden_states = th.cat([h_in.unsqueeze(0) for h_in, ag_out in h_in_agents_out])
        self.hidden_states = th.cat([a.hidden_state.unsqueeze(0) for a in self.agents])
        self.latent = th.cat([a.latent.unsqueeze(0) for a in self.agents])
        self.latent_infer = th.cat([a.latent_infer.unsqueeze(0) for a in self.agents])
        return agents_out

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        if self.policy_contain_mask:
            obs, acs, masks, rews, next_obs, dones = sample
        else:
            obs, acs, rews, next_obs, dones = sample
            masks = [None] * self.args.n_agents
        # Q loss
        next_acs = []
        next_log_pis = []
        self.bs = self.args.sample_size // self.args.episode_length
        self.init_hidden(self.bs)
        self.init_latent(self.bs)

        for ai, pi, mask, ob in zip(range(len(self.target_policies)), self.target_policies, masks, next_obs):
            curr_next_ac = []
            curr_next_log_pi = []
            h_in = self.hidden_states[ai]
            for t in range(self.args.episode_length):
                c_n_a, c_n_l_p = pi(ob.reshape(self.bs, self.args.episode_length, -1)[:, t, :],
                                                    h_in, t, mask=mask,  return_log_pi=True)
                h_in = pi.hidden_state
                curr_next_ac.append(c_n_a.unsqueeze(1))
                curr_next_log_pi.append(c_n_l_p.unsqueeze(1))
            next_acs.append(th.cat(curr_next_ac, dim=1).reshape(self.bs*self.args.episode_length, -1))
            next_log_pis.append(th.cat(curr_next_log_pi, dim=1).reshape(self.bs*self.args.episode_length, -1))
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.args.n_agents), next_qs,
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
            self.critic.parameters(), 10 * self.args.n_agents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        if self.policy_contain_mask:
            obs, acs, masks, rews, next_obs, dones = sample
        else:
            obs, acs, rews, next_obs, dones = sample
            masks = [None] * self.args.n_agents

        self.bs = self.args.sample_size // self.args.episode_length
        self.init_hidden(self.bs)
        self.init_latent(self.bs)

        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []
        latent, latent_sampled, latent_infer = [], [], []

        for a_i, pi, hidden_state, mask, ob in zip(range(self.args.n_agents), self.policies, self.hidden_states, masks, obs):
            curr_ac, probs, log_pi, pol_regs = [], [], [], []
            h_in  = self.hidden_states[a_i]
            l_ai, l_s_ai, l_i_ai = [], [], []
            for t in range(self.args.episode_length):
                c_a, p, l_p, p_r, e = pi(
                    ob.reshape(self.bs, self.args.episode_length, -1)[:, t, :],
                    h_in, t, mask, return_all_probs=True, return_log_pi=True,
                    regularize=True, return_entropy=True)
                h_in = pi.hidden_state
                l_ai.append(pi.latent.unsqueeze(1))
                l_i_ai.append(pi.latent_infer.unsqueeze(1))
                l_s_ai.append(pi.latent_sampled.unsqueeze(1))
                curr_ac.append(c_a.unsqueeze(1)); probs.append(p.unsqueeze(1)); log_pi.append(l_p.unsqueeze(1));
                pol_regs.append(p_r[0]);
            # self.hidden_states = th.cat([a.hidden_state.unsqueeze(0) for a in self.agents])
            l_ai = th.cat(l_ai, dim=1).reshape(self.bs*self.args.episode_length, self.latent_dim*2)
            l_i_ai = th.cat(l_i_ai, dim=1).reshape(self.bs*self.args.episode_length, self.latent_dim*2)
            l_s_ai = th.cat(l_s_ai, dim=1).reshape(self.bs*self.args.episode_length, self.latent_dim)
            #ent = [ent[t*i]  for i in range(self.bs) for t in range(len(ent))]
            ent = -(th.cat(log_pi, dim=1).reshape(-1, 1) * th.cat(probs, dim=1).reshape(-1, self.args.n_actions)).sum(1).mean()
            pol_regs = [th.tensor(pol_regs).mean()]
            if logger is not None:
                logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
                              self.niter)
            samp_acs.append(th.cat(curr_ac, dim=1).reshape(-1, self.args.n_actions))
            all_probs.append(th.cat(probs, dim=1).reshape(-1, self.args.n_actions))
            all_log_pis.append(th.cat(log_pi, dim=1).reshape(-1))
            all_pol_regs.append(pol_regs)
            latent.append(l_ai.unsqueeze(1))
            latent_sampled.append(l_s_ai.unsqueeze(1))
            latent_infer.append(l_i_ai.unsqueeze(1))

        latent = th.cat(latent, dim=1)
        latent_sampled = th.cat(latent_sampled, dim=1)
        latent_infer = th.cat(latent_infer, dim=1)
        #
        # inputs = inputs.reshape(-1, self.input_shape)
        #h_in = hidden_state.reshape(-1, self.hidden_dim)
        # if train_mode and (not self.args.roma_raw):
        latent = latent.reshape(self.bs * self.args.episode_length * self.n_agents, self.latent_dim * 2)
        latent_infer = latent_infer.reshape(self.bs * self.args.episode_length * self.n_agents, self.latent_dim * 2)
        gaussian_embed = D.Normal(latent[:, :self.latent_dim], (latent[:, self.latent_dim:]) ** (1 / 2))
        gaussian_infer = D.Normal(latent_infer[:, :self.latent_dim], (latent_infer[:, self.latent_dim:]) ** (1 / 2))

        loss = gaussian_embed.entropy().sum(dim=-1).mean() * self.args.h_loss_weight + kl_divergence(gaussian_embed, gaussian_infer).sum(dim=-1).mean() * self.args.kl_loss_weight   # CE = H + KL
        # ce_loss is Identifiable roles loss
        loss = th.clamp(loss, max=2e3)
        # loss = loss / (self.bs * self.n_agents)
        ce_loss = th.log(1 + th.exp(loss))   # Identifiable loss

        # Dis Loss
        cur_dis_loss_weight = self.dis_loss_weight_schedule(self.i_iter)
        if cur_dis_loss_weight > 0:
            dis_loss = 0
            dissimilarity_cat = None
            mi_cat = None
            latent_dis = latent_sampled.clone().view(self.bs * self.args.episode_length, self.n_agents, -1)
            latent_move = latent_sampled.clone().view(self.bs * self.args.episode_length, self.n_agents, -1)
            for agent_i in range(self.n_agents):
                latent_move = th.cat(
                    [latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1)
                latent_dis_pair = th.cat([latent_dis[:, :, :self.latent_dim],
                                          latent_move[:, :, :self.latent_dim],
                                          #(latent_dis[:, :, :self.latent_dim]-latent_move[:, :, :self.latent_dim])**2
                                          ], dim=2)

                # inference network? gaussian_embed is the role encoder
                mi = th.clamp(gaussian_embed.log_prob(latent_move.view(self.bs * self.args.episode_length * self.n_agents, -1))+13.9, min=-13.9).sum(dim=1,keepdim=True) / self.latent_dim

                dissimilarity = th.abs(self.dis_net(latent_dis_pair.view(-1, 2 * self.latent_dim)))

                if dissimilarity_cat is None:
                    dissimilarity_cat = dissimilarity.reshape(self.bs * self.args.episode_length, -1).clone()
                else:
                    dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(self.bs * self.args.episode_length, -1)], dim=1)
                if mi_cat is None:
                    mi_cat = mi.view(self.bs * self.args.episode_length, -1).clone()
                else:
                    mi_cat = th.cat([mi_cat,mi.view(self.bs * self.args.episode_length,-1)],dim=1)

                #dis_loss -= th.clamp(mi / 100 + dissimilarity, max=0.18).sum() / self.bs / self.n_agents

            mi_min=mi_cat.min(dim=1,keepdim=True)[0]
            mi_max=mi_cat.max(dim=1,keepdim=True)[0]
            di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
            di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]

            mi_cat=(mi_cat-mi_min)/(mi_max-mi_min+ 1e-12 )
            dissimilarity_cat=(dissimilarity_cat-di_min)/(di_max-di_min+ 1e-12 )

            dis_loss = - th.clamp(mi_cat+dissimilarity_cat, max=1.0).sum()/self.bs/self.n_agents/self.args.episode_length
            #dis_loss = ((mi_cat + dissimilarity_cat - 1.0 )**2).sum() / self.bs / self.n_agents
            dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum() / self.bs / self.n_agents/self.args.episode_length

            #c_dis_loss = (dis_loss + dis_norm) / self.n_agents * cur_dis_loss_weight

            # Specialized roles loss
            c_dis_loss = (dis_norm + self.args.soft_constraint_weight * dis_loss) / self.n_agents * cur_dis_loss_weight  # Specialized loss

            self.mi = mi_cat[0]
            self.dissimilarity = dissimilarity_cat[0]
        else:
            c_dis_loss =None # Specialized loss

        pol_losses = []
        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.args.n_agents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            #disable_gradients(self.critic)
            #pol_loss.backward()
            pol_losses += [pol_loss]
            #enable_gradients(self.critic)

        disable_gradients(self.critic)
        ce_loss.backward(retain_graph=True)#retain_graph=True)
        if c_dis_loss:
            c_dis_loss.backward(retain_graph=True)#retain_graph=True)
        enable_gradients(self.critic)

        for a_i in range(len(self.agents)):
            curr_agent = self.agents[a_i]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                curr_agent.policy.parameters(), self.args.clip)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_losses[a_i], self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)
        if logger is not None:
            if c_dis_loss:
                logger.add_scalar('losses/c_dis_loss',c_dis_loss, self.niter)
            logger.add_scalar('losses/ce_loss', ce_loss, self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def init_hidden(self, batch_size):
        if self.pol_dev == 'cuda':
            self.hidden_states = th.zeros(self.args.n_agents, batch_size,
                                          self.args.rnn_hidden_dim).cuda()  # (bs,n,hidden_dim)
            self.trgt_hidden_states = th.zeros(self.args.n_agents, batch_size,
                                          self.args.rnn_hidden_dim).cuda()  # (bs,n,hidden_dim)
        else:
            self.hidden_states = th.zeros(self.args.n_agents, batch_size, self.args.rnn_hidden_dim)
            self.trgt_hidden_states = th.zeros(self.args.n_agents, batch_size,  self.args.rnn_hidden_dim)

    def init_latent(self, batch_size, first_time=False):
        #self.bs = batch_size
        loss = 0

        if first_time:
            self.latent = th.rand(self.args.n_agents, 1, self.args.latent_dim * 2)  # (n,mu+var)
            self.mi = th.rand(self.args.n_agents * self.args.n_agents)
            self.dissimilarity = th.rand(self.args.n_agents * self.args.n_agents)

        self.trajectory = []

        var_mean = self.latent[:self.n_agents, :, self.args.latent_dim:].detach().mean()

        # mask = 1 - th.eye(self.n_agents).byte()
        # mi=self.mi.view(self.n_agents,self.n_agents).masked_select(mask)
        # di=self.dissimilarity.view(self.n_agents,self.n_agents).masked_select(mask)


        mi = self.mi.detach()
        di = self.dissimilarity.detach()
        indicator = [var_mean, mi.max(), mi.min(), mi.mean(), mi.std(), di.max(), di.min(), di.mean(), di.std()]
        return indicator #, self.latent[ :self.n_agents,:, :].detach(), self.latent_infer[ :self.n_agents,:, :].detach()

    def log_latent(self, logger = None, t_env = 0):
        indicator = self.init_latent(batch_size = self.bs)
        if logger:
        # indicator=[var_mean,mi.max(),mi.min(),mi.mean(),mi.std(),di.max(),di.min(),di.mean(),di.std()]
            logger.add_scalar("indicator/var_mean", indicator[0].item(), t_env)
            logger.add_scalar("indicator/mi_max", indicator[1].item(), t_env)
            logger.add_scalar("indicator/mi_min", indicator[2].item(), t_env)
            logger.add_scalar("indicator/mi_mean", indicator[3].item(), t_env)
            logger.add_scalar("indicator/mi_std", indicator[4].item(), t_env)
            logger.add_scalar("indicator/di_max", indicator[5].item(), t_env)
            logger.add_scalar("indicator/di_min", indicator[6].item(), t_env)
            logger.add_scalar("indicator/di_mean", indicator[7].item(), t_env)
            logger.add_scalar("indicator/di_std", indicator[8].item(), t_env)

            #self.logger.add_scalar("grad_norm", grad_norm, t_env)
            #mask_elems = mask.sum().item()
            #self.logger.add_scalar("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            #self.logger.add_scalar("q_taken_mean",
            #                     (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            #self.logger.add_scalar("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
            #                     t_env)

            #if self.args.use_tensorboard:
                # log_vec(self,mat,metadata,label_img,global_step,tag)
                #self.args.writer.log_vec(latent, list(range(self.args.n_agents)), t_env, "latent")
                #self.args.writer.log_vec(latent_vae, list(range(self.args.n_agents)), t_env, "latent-VAE")
            #self.log_stats_t = t_env

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

    def dis_loss_weight_schedule_step(self, t_glob):
        if t_glob > self.args.dis_time:
            return self.args.dis_loss_weight
        else:
            return 0

    def dis_loss_weight_schedule_sigmoid(self, t_glob):
        return self.args.dis_loss_weight / (1 + math.exp((1e7 - t_glob) / 2e6))

    def prep_rollouts(self, device='cpu'):
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
        writer = getattr(self.init_dict["args"], "writer", None)
        self.init_dict["args"].writer = None
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)
        self.init_dict["args"].writer = writer


    @classmethod
    def init_from_env(cls, env, mf_len=None, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, policy_contain_mask=False, observation_discrete=False,
                      args=None,
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

        if "n_agents" in vars(args):
            if len(env.action_space) != args.n_agents:
                print("WARNING:wrong number of agents in args, should be {} but got {}".format(len(env.action_space), args.n_agents))
        args.n_agents =  len(env.observation_space)

        if "n_actions" in vars(args):
            if len(env.action_space[0].n) != args.n_agctions:
                print("WARNING:wrong number of agents in args, should be {} but got {}".format(env.action_space[0].n, args.n_agctions))
        args.n_actions =  env.action_space[0].n

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size,
                     'policy_contain_mask': policy_contain_mask,
                     'args': args}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False, device='cpu'):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=device)
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