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
from xvfbwrapper import Xvfb
#%matplotlib inline
proj_direc = '/data/akvasov/bm3il/'
sys.path.append(os.path.abspath(os.path.join(
    proj_direc+'buildILGymEnvs', '..')))
from utils import *
from core.maagentMAACGymEnvs import AgentsInteraction
from algorithms.attention_sac import AttentionSAC
from utils.make_env import make_env
from utils.env_wrappers import StandardEnv
from utils.replay_memory_MAACGymEnvs import Memory


class ARGS():
    def __init__(self):
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
        self.gpu_index = 0
        self.seed = 1
        self.env_name = 'multi_speaker_listener' #'simple_spread' # 'fullobs_collect_treasure', 'multi_speaker_listener'
        # hyper-parameters to be tuned for only collecting exeprt trajectories
        self.reset_memory_interval = 10
        self.min_batch_size = 800 #4000
        self.sample_size = 800 # number of samples for an update
        self.num_threads = 8 #4
        self.generator_epochs = 10
        self.max_iter_num = 6000 #1500 #30 #500
        self.load_checkpoint = False
        self.save_checkpoint_interval = 100
        self.expert_traj_len = int(2e4) #5e4
        self.collect_expert_samples = 1600 #int(1e3)
        # save directories
        self.checkpoint_path = proj_direc + "buildILGymEnvs/data/" + self.env_name + "/checkpoint_maac2.tar"
        self.expert_traj_path = proj_direc + "buildILGymEnvs/data/" + self.env_name + "/exp_traj2.pkl"
        self.description = 'vanilaMAAC'
        self.save_data_path = proj_direc+"buildILGymEnvs/data/"+self.env_name+"_maac2_rewards.pkl"
        
        
args = ARGS()
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
    os.makedirs(os.path.abspath(os.path.join(args.expert_traj_path,'..')))
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

        
def update_params(batch, agentModels, agentsInteract):
    agentModels.prep_training(device=device.type)
    for _ in range(args.generator_epochs):
        sample = agentsInteract.batch2TensorSample(batch, cuda, sample_size=args.sample_size)
        agentModels.update_critic(sample)
        agentModels.update_policies(sample)
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
                                       reward_scale=args.reward_scale)
    if args.load_checkpoint:
        agentModels = AttentionSAC.init_from_save(args.checkpoint_path, load_critic=True)
    agentModels.prep_rollouts(device='cpu')
    agentsInteract = AgentsInteraction(env, numAgents, agentModels, device, running_state=None, render=args.render, num_threads=args.num_threads)
    time_list = list()
    rList = list()
    label_list = list()
    running_memory = [Memory() for _ in range(numAgents)]
    
    
    # train expert policy
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, _, log = agentsInteract.collect_samples(args.min_batch_size, args.episode_length, cuda, running_memory=running_memory)
        t0 = time.time()        
        update_params(batch, agentModels, agentsInteract)
        t1 = time.time()
        if (i_iter+1) % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_avg {:.2f}, {:.2f}, {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}\t running memory len {}'.format(
                i_iter, log['sample_time'], t1 - t0, np.mean(log['avg_reward']), log['avg_reward'][0], log['avg_reward'][1], log['num_episodes'], log['num_steps'], len(running_memory[0])))
        rList.append(log['avg_reward'])
        time_list.append(i_iter)
        label_list.append(args.description[0])
        # clean running memory
        if (i_iter+1) % args.reset_memory_interval == 0:
            for ai in range(numAgents):
                running_memory[ai].resetToLength(args.min_batch_size*(args.reset_memory_interval)//2)
        # save checkping
        if args.save_checkpoint_interval > 0 and (i_iter+1) % args.save_checkpoint_interval == 0:
            agentModels.save(args.checkpoint_path)
            agentModels.prep_rollouts(device='cpu')
            data_dic = {'time': time_list, 'reward': rList, 'Algorithms': label_list}
            df = pd.DataFrame(data_dic)
            df.to_pickle(args.save_data_path)
        """clean up gpu memory"""
        torch.cuda.empty_cache()
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
        exp_traj_df.to_pickle(expert_traj_path_ai)

        
main_loop()
#vdisplay.stop()
