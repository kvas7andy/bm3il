import argparse
import torch
import os
import gym
import gym_foo
import errno
import sys
import pickle
import time
import json
import numpy as np
import pandas as pd
#from xvfbwrapper import Xvfb
#%matplotlib inline
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

proj_direc = '/data/akvasov/bm3il/'
sys.path.append(os.path.abspath(os.path.join(
    proj_direc+'buildILGymEnvsRT', '..')))
from utils import *
from core.maagentMAACGymEnvs import AgentsInteraction
from algorithms.attention_sac import AttentionSAC
from utils.make_env import make_env
from utils.env_wrappers import StandardEnv
from utils.replay_memory_MAACGymEnvs import Memory

        
def update_params(batch, agentModels, agentsInteract):
    agentModels.prep_training(device=device.type)
    for _ in range(args.generator_epochs):
        sample = agentsInteract.batch2TensorSample(batch, cuda, sample_size=args.sample_size, norm_rews = True)
        agentModels.update_critic(sample, logger=writer)
        agentModels.update_policies(sample, logger=writer)
        agentModels.update_all_targets()
    sample = agentsInteract.batch2TensorSample(batch, cuda, sample_size=args.episode_length * 4,
                                               only_recent=True, random=False)
    agentModels.prep_rollouts(device=device.type)
    with torch.no_grad():
        ret_c = agentModels.value_critic(sample, logger=writer, episode_length=args.episode_length)
        ret_p = None
    #ret_p = agentModels.value_policies(sample, logger=writer)
    agentModels.prep_rollouts(device='cpu')
    return ret_c, ret_p

    
def main_loop(): 
    """create agent (including actor and critic)"""
    agentModels = AttentionSAC.init_from_env(env,
                                       tau=args.tau,
                                       pi_lr=args.pi_lr,
                                       q_lr=args.q_lr,
                                       gamma=args.gamma,
                                       pol_hidden_dim=args.pol_hidden_dim,
                                       critic_hidden_dim=args.critic_hidden_dim,
                                       attend_heads=args.attend_heads,
                                       reward_scale=args.reward_scale)
    if args.load_checkpoint:
        agentModels = AttentionSAC.init_from_save(args.checkpoint_path, load_critic=True)
    agentModels.prep_rollouts(device='cpu')
    agentsInteract = AgentsInteraction(env, numAgents, agentModels, device, running_state=None, render=args.render, num_threads=args.num_threads)
    time_list = list()
    avg_time_list = list()
    iter_list = list()
    reward_list = list()
    label_list = list()
    rList = list()
    rSampled = list()
    qSampled = list()
    tarSampled = list()
    running_memory = [Memory() for _ in range(numAgents)]

    max_reward = -1e10
    flush_flag = False
    t_start = time.time()
    # train expert policy
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, _, log = agentsInteract.collect_samples(args.min_batch_size, args.episode_length, cuda, running_memory=running_memory)
        t0 = time.time()        
        ret_c, ret_p = update_params(batch, agentModels, agentsInteract)
        t1 = time.time()
        avg_time_list.append(log['sample_time'] + t1 - t0)
        if (i_iter+1) % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_all {:.4f}\tT_run {:.4f}\tT_start {:.4f}\tR_avg {}\tEpisodes {:.2f}\tSteps {:.2f}\t running memory len {}'.format(
                i_iter, log['sample_time'], t1 - t0, avg_time_list[-1], np.mean(avg_time_list), t1-t_start, " ".join("{:.2f}".format(x) for x in log['avg_reward']), log['num_episodes'], log['num_steps'], len(running_memory[0])), flush=flush_flag)
        if flush_flag:
            flush_flag = False
        rList.append(log['avg_reward'])
        time_list.append(t1 - t_start)
        iter_list.append(i_iter)
        label_list.append(args.description[0])
        rSampled.append(ret_c[0])
        qSampled.append(ret_c[1])
        tarSampled.append(ret_c[2])
        # clean running memory
        if (i_iter+1) % args.reset_memory_interval == 0:
            for ai in range(numAgents):
                running_memory[ai].resetToLength(args.min_batch_size*(args.reset_memory_interval)//2)
        # save checkping
        if args.save_checkpoint_interval > 0 and (i_iter+1) % args.save_checkpoint_interval == 0:
            checkpoint_path_iter = args.checkpoint_path + '_iter' + str(i_iter) + '.tar'
            if args.save_or_not:
                agentModels.save(checkpoint_path_iter)
            if np.mean(log['avg_reward']) > max_reward:
                max_reward  = np.mean(log['avg_reward'])
                checkpoint_path_iter = args.checkpoint_path + 'best' + '.tar'
                if args.save_or_not:
                    agentModels.save(checkpoint_path_iter)
            agentModels.prep_rollouts(device='cpu')
            data_dic = {'iter': iter_list, 'time': time_list, 'reward': rList, 'Algorithms': label_list}
            reward_dic = {'iter': iter_list}
            for ai in range(numAgents):
                reward_dic.update({'agent%i_rews' % ai: rSampled,
                                   'agent%i_qvals' % ai: qSampled,
                                   'agent%i_TDtarvals' % ai: tarSampled})
            df = pd.DataFrame(data_dic)
            df_r = pd.DataFrame(reward_dic)
            if args.save_or_not:
                df.to_pickle(args.save_data_path)
                df_r.to_pickle(args.save_data_path.split('.pkl')[0] + '_rewards.pkl')
        """clean up gpu memory"""
        torch.cuda.empty_cache()
        if args.save_checkpoint_interval > 0 and (i_iter + 1) % (args.save_checkpoint_interval/10) == 0:
            flush_flag = True
            if writer is not None:
                writer.flush()


    # save training epoch
    rLarge = np.array([np.average(x) for x in rList[-10:]])
    rMean = np.mean(rLarge[np.argsort(rLarge)[-5:]])
    rStd = np.abs(rMean)
    
    
    
    # collect expert traj
    print('collect expert trajectories')
    qualify_states = [np.zeros((1,env.observation_space[ai].shape[0])) for ai in range(numAgents)]
    qualify_actions = [np.zeros((1, env.action_space[ai].n)) for ai in range(numAgents)]
    qualify_rewards = [np.zeros((1)) for _ in range(numAgents)]
    while qualify_states[0].shape[0] < args.expert_traj_len:
        batch, _, log = agentsInteract.collect_samples(args.collect_expert_samples, args.episode_length, cuda)
        dones = np.stack(batch[0].dones)
        r_mean = np.mean(log['reward_list'])
        r_std = np.std(log['reward_list'])
        r_dif = max(np.mean(log['max_reward']) - r_mean, r_mean - np.mean(log['min_reward']))
        print(
            'Traj_len {}\tConstrain_rMean {:.2f}\trBound {:.2f}\tR_avg {:.2f} +- {:.2f}\t R_std {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}'.format(
                qualify_states[0].shape[0],rMean,rStd, r_mean, r_dif, r_std, \
                log['num_episodes'], log['num_steps']))
        reward_list = [np.average(x) for x in log['reward_list']]
        qualify_index = [r < rMean + rStd and r > rMean - rStd for r in reward_list]
        #start_index = np.concatenate(([0], np.where(dones == 1)[0] + 1))
        start_index = np.arange(len(batch[0][0]),step=args.episode_length)
        for ai in range(numAgents):
            states = np.stack(batch[ai].state)
            actions = np.stack(batch[ai].action)
            if len(actions.shape) == 1:
                actions = np.expand_dims(actions,-1)
            qualify_states[ai] = np.concatenate((qualify_states[ai], states), axis=0)
            qualify_actions[ai] = np.concatenate((qualify_actions[ai], actions), axis=0)
            qualify_rewards[ai] = np.concatenate((qualify_rewards[ai], reward_list),axis = 0)

                    
    # save expert traj
    for ai in range(numAgents):
        qualify_states[ai] = qualify_states[ai][1:,:]
        qualify_actions[ai] = qualify_actions[ai][1:,:]
        qualify_rewards[ai] = qualify_rewards[ai][1:]
    rUp = np.asarray(qualify_rewards).max()-np.asarray(qualify_rewards).mean()
    rBot = np.asarray(qualify_rewards).mean()-np.asarray(qualify_rewards).min()
    rBound = np.max((rUp,rBot))
    print('Constrain rMean {:.2f}, rBound {:.2f}, result rMean {:.2f}, rBound {:.2f}, rStd {:.2f}, traj len {}'\
          .format(rMean,rStd,np.mean(np.asarray(qualify_rewards)),rBound,np.std(np.asarray(qualify_rewards)),qualify_states[0].shape[0]))
    for ai in range(numAgents):
        if len(qualify_actions[ai].shape)<len(qualify_states[ai].shape):
            qualify_actions[ai] = np.expand_dims(qualify_actions[ai],-1)
        exp_traj = np.concatenate((qualify_states[ai],qualify_actions[ai]),axis=1)
        expert_traj_path_ai = args.expert_traj_path + '_agent_' + str(ai)
        exp_traj_df = pd.DataFrame(exp_traj)
        if args.save_or_not:
            exp_traj_df.to_pickle(expert_traj_path_ai)


def custom_model_loop():
    """create agent (including actor and critic)"""
    agentModels = AttentionSAC.init_from_env(env,
                                             tau=args.tau,
                                             pi_lr=args.pi_lr,
                                             q_lr=args.q_lr,
                                             gamma=args.gamma,
                                             pol_hidden_dim=args.pol_hidden_dim,
                                             critic_hidden_dim=args.critic_hidden_dim,
                                             attend_heads=args.attend_heads,
                                             reward_scale=args.reward_scale)
    if args.load_checkpoint:
        agentModels = AttentionSAC.init_from_save(args.checkpoint_path, load_critic=True)
    agentModels.prep_rollouts(device='cpu')
    agentsInteract = AgentsInteraction(env, numAgents, agentModels, device, running_state=None, render=args.render,
                                       num_threads=args.num_threads)
    rList = list()

    # save training epoch
    rLarge = np.array([np.average(x) for x in rList[-10:]])
    rMean = np.mean(rLarge[np.argsort(rLarge)[-5:]])
    rStd = np.abs(rMean)

    # collect expert traj
    print('collect expert trajectories')
    qualify_states = [np.zeros((1, env.observation_space[ai].shape[0])) for ai in range(numAgents)]
    qualify_actions = [np.zeros((1, env.action_space[ai].n)) for ai in range(numAgents)]
    qualify_rewards = [np.zeros((1)) for _ in range(numAgents)]
    while qualify_states[0].shape[0] < args.expert_traj_len:
        batch, _, log = agentsInteract.collect_samples(args.collect_expert_samples, args.episode_length, cuda)
        dones = np.stack(batch[0].dones)
        r_mean = np.mean(log['reward_list'])
        r_std = np.std(log['reward_list'])
        r_dif = max(np.mean(log['max_reward']) - r_mean, r_mean - np.mean(log['min_reward']))
        print(
            'Traj_len {}\tConstrain_rMean {:.2f}\trBound {:.2f}\tR_avg {:.2f} +- {:.2f}\t R_std {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}'.format(
                qualify_states[0].shape[0], rMean, rStd, r_mean, r_dif, r_std, \
                log['num_episodes'], log['num_steps']))
        reward_list = [np.average(x) for x in log['reward_list']]
        qualify_index = [r < rMean + rStd and r > rMean - rStd for r in reward_list]
        # start_index = np.concatenate(([0], np.where(dones == 1)[0] + 1))
        start_index = np.arange(len(batch[0][0]), step=args.episode_length)
        for ai in range(numAgents):
            states = np.stack(batch[ai].state)
            actions = np.stack(batch[ai].action)
            if len(actions.shape) == 1:
                actions = np.expand_dims(actions, -1)
            qualify_states[ai] = np.concatenate((qualify_states[ai], states), axis=0)
            qualify_actions[ai] = np.concatenate((qualify_actions[ai], actions), axis=0)
            qualify_rewards[ai] = np.concatenate((qualify_rewards[ai], reward_list), axis=0)

    # save expert traj
    for ai in range(numAgents):
        qualify_states[ai] = qualify_states[ai][1:, :]
        qualify_actions[ai] = qualify_actions[ai][1:, :]
        qualify_rewards[ai] = qualify_rewards[ai][1:]
    rUp = np.asarray(qualify_rewards).max() - np.asarray(qualify_rewards).mean()
    rBot = np.asarray(qualify_rewards).mean() - np.asarray(qualify_rewards).min()
    rBound = np.max((rUp, rBot))
    print('Constrain rMean {:.2f}, rBound {:.2f}, result rMean {:.2f}, rBound {:.2f}, rStd {:.2f}, traj len {}' \
          .format(rMean, rStd, np.mean(np.asarray(qualify_rewards)), rBound, np.std(np.asarray(qualify_rewards)),
                  qualify_states[0].shape[0]))
    for ai in range(numAgents):
        if len(qualify_actions[ai].shape) < len(qualify_states[ai].shape):
            qualify_actions[ai] = np.expand_dims(qualify_actions[ai], -1)
        exp_traj = np.concatenate((qualify_states[ai], qualify_actions[ai]), axis=1)
        expert_traj_path_ai = args.expert_traj_path + '_agent_' + str(ai)
        exp_traj_df = pd.DataFrame(exp_traj)
        if args.save_or_not:
            exp_traj_df.to_pickle(expert_traj_path_ai)


class ARGS():
    def __init__(self, config=None):
        # hyper-parameters for MAAC
        self.pol_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.attend_heads = 4
        self.pi_lr = 1e-3 #1e-3
        self.q_lr = 1e-3 #1e-3
        self.tau = 1e-3 #1e-3
        self.gamma = 0.99
        self.reward_scale = 100
        self.episode_length = 25 #100, 25
        # hyper-parameters for IL
        self.render = False #False
        self.log_interval = 1
        self.gpu_index = 2
        self.seed = 1
        # environment params
        self.env_name = config.env_name if config is not None and hasattr(config, 'env_name') and config.env_name is not None \
            else 'simple_spread' #'simple_spread'#'multi_speaker_listener_2' #'simple_spread' # 'fullobs_collect_treasure', 'multi_speaker_listener'
        # hyper-parameters to be tuned for only collecting expert trajectories
        self.reset_memory_interval = 10
        self.min_batch_size = 800 #4000
        self.sample_size = 800 # number of samples for an update
        self.num_threads = 8 #4
        self.generator_epochs = 10
        self.max_iter_num = 6000 #1500 #30 #500 #3000
        self.load_checkpoint = False
        self.save_checkpoint_interval = 500
        self.expert_traj_len = int(5e4) #5e4
        self.collect_expert_samples = 1600 #int(1e3)
        # save directories
        dt_string = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.exper_path = os.path.join(proj_direc, "buildILGymEnvsRT/data", self.env_name, "collect_samples", dt_string)
        self.checkpoint_path = os.path.join(self.exper_path, "checkpoint_vanilaMAAC")
        self.expert_traj_path = os.path.join(self.exper_path, "exp_traj2" + ".pkl")
        self.description = 'vanilaMAAC'
        self.save_data_path = os.path.join(self.exper_path, self.env_name + "_vanilaMAAC.pkl")
        self.save_or_not = config.save_true if config is not None and hasattr(config, 'save_true') else False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help="Name of environment", type=str)
    parser.add_argument("--model_name", default='maac',
                        help="Name of directory to store " +
                             "model/training contents", type=str)
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--save_true", action='store_true')

    config = parser.parse_args()
    args = ARGS(config)
    dtype = torch.float
    torch.set_default_dtype(dtype)
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda', index=args.gpu_index) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(args.gpu_index)
    """environment"""
    rawEnv = make_env(args.env_name, discrete_action=True)
    env = StandardEnv(rawEnv)
    numAgents = len(env.observation_space)
    # state_dim = env.observation_space[0].shape[0]
    # is_disc_action = len(env.action_space[0].shape) == 0
    # action_dim = env.action_space[0].n if is_disc_action else env.action_space.shape[0]
    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    """create save directory"""
    try:
        os.makedirs(os.path.abspath(os.path.join(args.expert_traj_path, '..')), exist_ok=True)
        if args.save_or_not:
            os.makedirs(args.exper_path, exist_ok=True)
            os.makedirs(os.path.join(args.exper_path, "logs"), exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    """redirecting output"""
    writer = None
    if args.save_or_not:
        sys.stdout = open(os.path.join(args.exper_path, "logs", "log.txt"), 'w')
        writer = SummaryWriter(os.path.join(args.exper_path, "logs"))
        print(vars(args))
        print(" ".join(sys.argv))

    if config.model_name == 'maac':
        main_loop()
    else:
        custom_model_loop(config.model_name)
    # vdisplay.stop()



