#import torch
import os
#import gym
#import gym_foo
import errno
#import sys
#import pickle
import time
#import json
#import numpy as np
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

from algorithms.attention_sac_latent import AttentionSACLatent
import utils.config as config


class ARGS():
    def __init__(self, kwargs):
        # hyper-parameters for MAAC
        self.rnn_true = True
        self.pol_hidden_dim = 128 # 128
        self.critic_hidden_dim = 128 # 128
        self.attend_heads = 4
        self.pi_lr = 1e-3 #1e-3
        self.q_lr = 1e-3 #1e-3
        self.tau = 1e-3 #1e-3
        self.gamma = 0.99
        self.reward_scale = 100
        self.episode_length = 50 #100, 25
        # hyper-parameters for IL
        self.render = False #False
        self.log_interval = 1
        self.gpu_index = 2
        self.seed = 1 #np.int(time.time())
        self.env_name = 'diverse_spread_v1' #'simple_spread' # 'fullobs_collect_treasure', 'multi_speaker_listener'
        # hyper-parameters to be tuned for only collecting exeprt trajectories
        self.reset_memory_interval = 10
        self.min_batch_size = 800 #4000
        self.sample_size = 800 # number of samples for an update
        self.num_threads = 8 #4
        self.generator_epochs = 10
        self.max_iter_num = 4000 #1500 #30 #500 #6000
        self.load_checkpoint = False
        self.save_checkpoint_interval = 100
        self.expert_traj_len = int(2e4) #5e4
        self.collect_expert_samples = 1600 #int(1e3)
        #
        for key, value in kwargs.items():
            if getattr(self, key, None) is not None:
                print("WARNING: There is already parameter with name ", key)
            if type(value) is type({}):
                setattr(self, key, config.obj(value))
            else:
                setattr(self, key, value)

        # save directories
        dt_string = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.exper_path = os.path.join(proj_direc, "buildILGymEnvsRT/data", self.env_name, "collect_samples",
                                       ("latent_" if getattr(self, "latent_true", False) else "") + "MAA2C_" + dt_string)
        self.checkpoint_path = os.path.join(self.exper_path, "checkpoint_vanilaMAAC")
        self.expert_traj_path = os.path.join(self.exper_path, "exp_traj2" + \
                                             ("_2" if "_2" in self.env_name else "") + ".pkl")
        self.load_checkpoint_path=os.path.join(proj_direc, "buildILGymEnvsRT/data", self.env_name, "collect_samples",
                                               '16_05_2021_00:53:13', "checkpoint_vanilaMAAC") + '_iter' + '2999' + '.tar'
        self.description = 'vanilaMAAC'
        self.save_data_path = os.path.join(self.exper_path, self.env_name + "_vanilaMAAC_rewards.pkl")
        self.save_or_not = True
        
params = deepcopy(sys.argv)
config_dict = config.create_config_dict(params)
#config_obj = config.obj(config_dict)
args = ARGS(config_dict)
add_params = {}
if "latent_true" in vars(args):
    AttentionSAC = AttentionSACLatent
    add_params = {'args': args}
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
#state_dim = env.observation_space[0].shape[0]
#is_disc_action = len(env.action_space[0].shape) == 0
#action_dim = env.action_space[0].n if is_disc_action else env.action_space.shape[0]
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)
"""create save directory"""
try:
    os.makedirs(os.path.abspath(os.path.join(args.expert_traj_path,'..')), exist_ok=True)
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
args.writer = writer
print(vars(args))

def update_params(batch, agentModels, agentsInteract):
    agentModels.prep_training(device=device.type)
    for _ in range(args.generator_epochs):
        episode_l_sample = args.episode_length if args.rnn_true else None
        sample = agentsInteract.batch2TensorSample(batch, cuda, sample_size=args.sample_size, episode_l_sample=episode_l_sample)
        #print("Sample size from args {} and from drawn samples {} ".format(args.sample_size, sample[0][0].shape[0]))
        agentModels.update_critic(sample, logger=writer)
        agentModels.update_policies(sample, logger=writer)
        agentModels.update_all_targets()
    agentModels.prep_rollouts(device='cpu')

    
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
                                   reward_scale=args.reward_scale,
                                   **add_params)
    offset = 0
    if args.load_checkpoint:
        # if "latent" not in var(args):
        agentModels = AttentionSAC.init_from_save(args.load_checkpoint_path, load_critic=True, device=device, **add_params)
        # else:
        offset = np.int(args.load_checkpoint_path.split('.')[0].split('iter')[1])+1
        print(offset)
    agentModels.prep_rollouts(device='cpu')
    agentsInteract = AgentsInteraction(env, numAgents, agentModels, device, running_state=None, render=args.render, num_threads=args.num_threads)
    time_list = list()
    avg_time_list = list()
    iter_list = list()
    reward_list = list()
    label_list = list()
    rList = list()
    running_memory = [Memory() for _ in range(numAgents)]

    max_reward = -1e10
    flush_flag = False
    t_start = time.time()
    # train expert policy
    for i_iter in range(offset, offset + args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        agentModels.i_iter = i_iter
        batch, _, log = agentsInteract.collect_samples(args.min_batch_size, args.episode_length, cuda, running_memory=running_memory)
        t0 = time.time()
        update_params(batch, agentModels, agentsInteract)
        t1 = time.time()
        avg_time_list.append(log['sample_time'] + t1 - t0)
        if (i_iter+1) % args.log_interval == 0:
            if getattr(agentModels, "log_latent", False):
                agentModels.log_latent(logger=writer, t_env=agentModels.niter)
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_all {:.4f}\tT_run {:.4f}\tT_start {:.4f}\tR_avg {:.2f}, {:.2f}, {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}\t running memory len {}'.format(
                i_iter, log['sample_time'], t1 - t0, avg_time_list[-1], np.mean(avg_time_list), t1-t_start, np.mean(log['avg_reward']), log['avg_reward'][0], log['avg_reward'][1], log['num_episodes'], log['num_steps'], len(running_memory[0])), flush=flush_flag)
        if flush_flag:
            flush_flag = False
        rList.append(log['avg_reward'])
        time_list.append(t1 - t_start)
        iter_list = list().append(i_iter)
        label_list.append(args.description[0])
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
                checkpoint_path_iter = args.checkpoint_path + '_bestiter' + str(i_iter) +'.tar'
                if args.save_or_not:
                    agentModels.save(checkpoint_path_iter)
            agentModels.prep_rollouts(device='cpu')
            data_dic = {'iter': iter_list, 'time': time_list, 'reward': rList, 'Algorithms': label_list}
            df = pd.DataFrame(data_dic)
            if args.save_or_not:
                df.to_pickle(args.save_data_path)
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

        
main_loop()
#vdisplay.stop()
