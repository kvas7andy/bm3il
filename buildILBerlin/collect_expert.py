import torch
import os
import gym
import errno
import sys
import pickle
import time
import json
import numpy as np
import pandas as pd
proj_direc = '/data/akvasov/bm3il/'
sys.path.append(os.path.abspath(os.path.join(
    proj_direc+'buildILBerlin', '..')))
from utils import *
from core.maagentMAAC import AgentsInteraction
from algorithms.attention_sac_cm import AttentionSAC


class ARGS():
    def __init__(self):
        # hyper-parameters for MAAC
        self.pol_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.attend_heads = 4
        self.pi_lr = 3e-3 #2e-3
        self.q_lr = 3e-3 #2e-3
        self.tau = 2e-3
        self.gamma = 0.99
        self.reward_scale = 100
        # hyper-parameters for IL
        self.render = False
        self.log_interval = 1
        self.gpu_index = 0
        self.seed = 1
        self.env_name = 'berlin-46-2-type10-v3'
        # mean field
        self.meanField = True
        # hyper-parameters to be tuned for only collecting exeprt trajectories
        self.min_batch_size = 1000 #2000
        self.num_threads = 8 #4
        self.generator_epochs = 10
        self.max_iter_num = 1200 #1500 #30 #500
        self.load_checkpoint = False
        self.save_checkpoint_interval = 100
        self.expert_traj_len = int(2e4) #5e4
        self.collect_expert_samples = int(2e3)
        # save directories
        self.checkpoint_path = proj_direc + "buildILBerlin/data/" + self.env_name + "/checkpoint_maac2_2_cm_mf_t10.tar"
        self.expert_traj_path = proj_direc + "buildILBerlin/data/" + self.env_name + "/exp_traj2_2_cm_mf_t10.pkl"
        self.description = 'vanilaMAAC'
        self.save_data_path = proj_direc+"buildILBerlin/data/"+self.env_name+"_maac2_2_cm_mf_t10_rewards.pkl"
        self.save_vehicle_location_path = proj_direc+"buildILBerlin/data/"+self.env_name+"_maac2_2_cm_mf_t10_vehicle_location.pkl"
        self.save_lane_vehicles_path = proj_direc+"buildILBerlin/data/"+self.env_name+"_maac2_2_cm_mf_t10_lane_vehicles.pkl"
        
        
args = ARGS()
dtype = torch.float
torch.set_default_dtype(dtype)
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda', index=args.gpu_index) if cuda else torch.device('cpu')
if cuda:
    torch.cuda.set_device(args.gpu_index)
"""environment"""
env = gym.make(args.env_name)
numAgents = env.numAgents
state_dim = env.observation_space[0].n
is_disc_action = len(env.action_space[0].shape) == 0
action_dim = env.action_space[0].n if is_disc_action else env.action_space.shape[0]
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

        
def update_params(sample, agentModels):
    agentModels.prep_training(device=device.type)
    for _ in range(args.generator_epochs):
        agentModels.update_critic(sample)
        agentModels.update_policies(sample)
        agentModels.update_all_targets()
    agentModels.prep_rollouts(device='cpu')

    
def main_loop(): 
    """create agent (including actor and critic)"""
    agentModels = AttentionSAC.init_from_env(env,
                                       mf_len=[env.observation_space[0].n for _ in range(len(env.agent2type))],
                                       tau=args.tau,
                                       pi_lr=args.pi_lr,
                                       q_lr=args.q_lr,
                                       gamma=args.gamma,
                                       pol_hidden_dim=args.pol_hidden_dim,
                                       critic_hidden_dim=args.critic_hidden_dim,
                                       attend_heads=args.attend_heads,
                                       policy_contain_mask=True,
                                       observation_discrete=True, 
                                       reward_scale=args.reward_scale)
    if args.load_checkpoint:
        agentModels = AttentionSAC.init_from_save(args.checkpoint_path, load_critic=True)
    agentModels.prep_rollouts(device='cpu')
    agentsInteract = AgentsInteraction(env, numAgents, agentModels, device, running_state=None, render=args.render, num_threads=args.num_threads, meanField=args.meanField, type2agent=env.type2agent, agent2type=env.agent2type)
    time_list = list()
    rList = list()
    label_list = list()
    
    
    # train expert policy
    for i_iter in range(args.max_iter_num+1):
        """generate multiple trajectories that reach the minimum batch_size"""
        _, sample, log = agentsInteract.collect_samples(args.min_batch_size, cuda)
        t0 = time.time()        
        update_params(sample, agentModels)
        t1 = time.time()
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_avg {:.2f}, {:.2f}, {:.2f}, {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, np.mean(log['avg_reward']), log['avg_reward'][0], log['avg_reward'][1], log['avg_reward'][2], log['num_episodes'], log['num_steps']))
        rList.append(log['avg_reward'])
        time_list.append(i_iter)
        label_list.append(args.description[0])
        # save checkping
        if args.save_checkpoint_interval > 0 and i_iter % args.save_checkpoint_interval == 0:
            agentModels.save(args.checkpoint_path)
            agentModels.prep_rollouts(device='cpu')
            data_dic = {'time': time_list, 'reward': rList, 'Algorithms': label_list}
            df = pd.DataFrame(data_dic)
            df.to_pickle(args.save_data_path)
            df_vehicle = pd.DataFrame(env.xt_est)
            df_vehicle.to_pickle(args.save_lane_vehicles_path)
            df_vehicle = pd.DataFrame(env.st_est)
            df_vehicle.to_pickle(args.save_vehicle_location_path)
        """clean up gpu memory"""
        torch.cuda.empty_cache()
    rLarge = np.array([np.average(x) for x in rList[-10:]])
    rMean = np.mean(rLarge[np.argsort(rLarge)[-5:]])
    rStd = np.abs(rMean/2)
    
    
    # collect expert traj
    print('collect expert trajectories')
    qualify_states = [np.zeros((1,state_dim*2)) for _ in range(numAgents)]
    qualify_actions = [np.zeros((1, action_dim)) for _ in range(numAgents)]
    qualify_rewards = [np.zeros((1)) for _ in range(numAgents)]
    while qualify_states[0].shape[0] < args.expert_traj_len:
        batch, _, log = agentsInteract.collect_samples(args.collect_expert_samples, cuda)
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
        start_index = np.concatenate(([0], np.where(dones == 1)[0] + 1))
        for i, s in enumerate(start_index[:-1]):
            if qualify_index[i]:
                e = start_index[i + 1]
                for ai in range(numAgents):
                    states = np.stack(batch[ai].state)
                    actions = np.stack(batch[ai].action)
                    if len(actions.shape) == 1:
                        actions = np.expand_dims(actions,-1)
                    qualify_states[ai] = np.concatenate((qualify_states[ai], states[s:e, :]), axis=0)
                    qualify_actions[ai] = np.concatenate((qualify_actions[ai], actions[s:e, :]), axis=0)
                    qualify_rewards[ai] = np.concatenate((qualify_rewards[ai], np.expand_dims(reward_list[i],-1)),axis = 0)

                    
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
