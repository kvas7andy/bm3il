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
from core.maagentMAACMeanField import AgentsInteraction
from algorithms.attention_sac_cm_cmf import AttentionSAC
from models.mlp_discriminator import Discriminator
from utils.svgd import RBF, phi

class ARGS():
    def __init__(self):
        # hyper-parameters for MAAC
        self.pol_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.discrim_net_nums = 4
        self.attend_heads = 4
        self.pi_lr = 3e-3 #3e-3
        self.q_lr = 3e-3 #3e-3
        # target network learn rate
        self.tau = 2e-3 #3e-3
        self.gamma = 0.99
        self.reward_scale = 100
        # hyper-parameters for IL
        self.render = False
        self.log_interval = 1
        self.gpu_index = 0 #0
        self.seed = 1
        self.env_name = 'berlin-46-2-type10-v3'
        self.expert_traj_path = proj_direc + "buildILBerlin/data/" + self.env_name + "/exp_traj2_2_cm_mf_t10.pkl"
        # mean field
        self.meanField = True
        self.mfApproxVersion = 2
        # hyper-parameters for MAIL
        self.expert_traj_len = int(1e4)
        self.min_batch_size = 1000
        self.num_threads = 8 #4
        self.epochs = 6
        self.d_lr = 2e-3
        self.discriminator_epochs = 3
        self.generator_epochs = 10
        self.max_iter_num = 1500 #1500
        self.load_checkpoint = False
        self.save_checkpoint_interval = 100
        # GMMIL
        self.sigma_list = [sigma / 1.0 for sigma in [1, 2, 4, 8, 16]]
        # save directories
        self.checkpoint_path = proj_direc + "buildILBerlinGAIL/data/" + self.env_name + "/checkpoint_bayesianGAIL3_cm2_mf_ds"
        self.description = 'bayesianGAIL'
        self.save_data_path = proj_direc+"buildILBerlinGAIL/data/"+self.env_name+"_bayesianGAIL3_cm2_mf_ds.pkl"
        self.save_vehicle_location_path = proj_direc+"buildILBerlinGAIL/data/"+self.env_name+"_bayesianGAIL3_cm2_mf_ds_vehicle_location"
        self.save_lane_vehicles_path = proj_direc+"buildILBerlinGAIL/data/"+self.env_name+"_bayesianGAIL3_cm2_mf_ds_lane_vehicles"


args = ARGS()
dtype = torch.float
torch.set_default_dtype(dtype)
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda', index=args.gpu_index) if cuda else torch.device('cpu')
if cuda:
    torch.cuda.set_device(args.gpu_index)
discrim_criterion = torch.nn.BCELoss()
to_device(device, discrim_criterion)
"""environment"""
env = gym.make(args.env_name)
numAgents = env.numAgents
is_disc_action = len(env.action_space[0].shape) == 0
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)
"""create save directory"""
try:
    os.makedirs(os.path.abspath(os.path.join(args.checkpoint_path,'..')))
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
"""load expert trajectory"""
expert_traj = list()
expert_traj_s = list()
expert_traj_a = list()
listMean = lambda x: np.array([sum(y)/len(y) for y in zip(*x)])
for ai in range(numAgents):
    expert_traj_path_ai = args.expert_traj_path + '_agent_' + str(ai)
    expert_traj_ai = pd.read_pickle(expert_traj_path_ai).to_numpy()
    expert_traj_ai.dtype='float'
    traj_s = expert_traj_ai[:args.expert_traj_len,:env.observation_space[ai].n]
    traj_a = expert_traj_ai[:args.expert_traj_len,-env.action_space[ai].n:]
    expert_traj_s.append(traj_s)
    expert_traj_a.append(traj_a)
expert_traj = [np.concatenate((expert_traj_s[ai],expert_traj_a[ai]),1) for ai in range(numAgents)]

class DiscriminatorWrap:
    def __init__(self,numAgents):
        self.numAgents = numAgents
        self.discrimNets = []
        self.optimizerDiscrims = []


"""create agent (including actor and critic)"""
def create_networks():
    agentModels = AttentionSAC.init_from_env(env,
                                       mf_len=[env.action_space[0].n for _ in range(len(env.agent2type))],
                                       tau=args.tau,
                                       pi_lr=args.pi_lr,
                                       q_lr=args.q_lr,
                                       gamma=args.gamma,
                                       pol_hidden_dim=args.pol_hidden_dim,
                                       critic_hidden_dim=args.critic_hidden_dim,
                                       attend_heads=args.attend_heads,
                                       policy_contain_mask=True,
                                       observation_discrete=True,
                                       mfApproxVersion=args.mfApproxVersion,
                                       reward_scale=args.reward_scale)
    if args.load_checkpoint:
        checkpoint_path_epoch = args.checkpoint_path + '_epoch' + str(0) + '.tar'
        agentModels = AttentionSAC.init_from_save(checkpoint_path_epoch, load_critic=True)
    agentModels.prep_rollouts(device='cpu')
    discrimListBayesian = list()
    for _ in range(args.discrim_net_nums):
        discrimList = DiscriminatorWrap(numAgents)
        for ai in range(numAgents):
            discrim_net = Discriminator(env.observation_space[ai].n + env.action_space[ai].n, hidden_size=(128,64), activation='relu')
            discrim_net.to(device)
            optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.d_lr)
            discrimList.discrimNets.append(discrim_net)
            discrimList.optimizerDiscrims.append(optimizer_discrim)
        discrimListBayesian.append(discrimList)
    if args.load_checkpoint:
        checkpointD_path_epoch = args.checkpoint_path + 'D_epoch' + str(0) + '.tar'
        checkpoint = torch.load(checkpointD_path_epoch)
        for k,v in checkpoint.items():
            kName,kD,kIdx = k.split('_')
            if kName=='discrimNets':
                discrimListBayesian[int(kD)].discrimNets[int(kIdx)].load_state_dict(v)
            elif kName=='optimizerDiscrims':
                discrimListBayesian[int(kD)].optimizerDiscrims[int(kIdx)].load_state_dict(v)
    agentsInteract = AgentsInteraction(env, numAgents, agentModels, device, running_state=None, render=args.render, num_threads=args.num_threads, type2agent=env.type2agent, agent2type=env.agent2type)
    return agentModels, discrimListBayesian, agentsInteract


def update_params(sample, agentModels, discrimListBayesian):
    """IRL estimate reward"""
    #obs, acs, mfs, masks, rews, next_obs, dones = sample
    dataSize = sample[0][0].shape[0]
    exp_idx = [random.sample(range(expert_traj[0].shape[0]), dataSize) for _ in range(args.discriminator_epochs)]
    for ai in range(numAgents):
        rewards = list()
        dis_input_fake = torch.cat((sample[0][ai],sample[1][ai]),dim=1)
        for di in range(args.discrim_net_nums):
            g_o = discrimListBayesian[di].discrimNets[ai](dis_input_fake)
            rewards.append(-torch.log(g_o).to(dtype).to(device).detach()/args.discrim_net_nums)
        # use this reward -> il, comment thie line -> rl
        sample[4][ai] = torch.stack(rewards, dim=0).sum(dim=0)
        
        """update discriminator"""
        for itDisc in range(args.discriminator_epochs):
            dis_input_real = torch.from_numpy(expert_traj[ai][exp_idx[itDisc], :]).to(dtype).to(device)
            
            
            # svgd
            dataDict = [[] for _ in range(args.discrim_net_nums)]
            gradDict = [[] for _ in range(args.discrim_net_nums)]
            for di in range(args.discrim_net_nums):
                g_o = discrimListBayesian[di].discrimNets[ai](dis_input_fake)
                e_o = discrimListBayesian[di].discrimNets[ai](dis_input_real)
                discrimListBayesian[di].optimizerDiscrims[ai].zero_grad()
                discrim_loss = discrim_criterion(g_o, ones((g_o.shape[0], 1), device=device)) + \
                               discrim_criterion(e_o, zeros((e_o.shape[0], 1), device=device))
                discrim_loss.backward()
                for param in discrimListBayesian[di].discrimNets[ai].parameters():
                    dataDict[di].append(param.data)
                    gradDict[di].append(param.grad)
            for idx, (w, grad) in enumerate(zip(zip(*dataDict), zip(*gradDict))):
                X = torch.stack(([wc.view(-1) for wc in w]),dim=0)
                score_func = torch.stack(([gc.view(-1) for gc in grad]),dim=0)
                svgdGrad = phi(score_func, RBF, X)
                for di in range(args.discrim_net_nums):
                    gradDict[di][idx] = svgdGrad[di].view(grad[di].shape).data
            for di in range(args.discrim_net_nums):
                for param_cur, param_best in zip(discrimListBayesian[di].discrimNets[ai].parameters(), gradDict[di]):
                    param_cur.grad = param_best
                discrimListBayesian[di].optimizerDiscrims[ai].step()

        
    """RL learn policy"""
    agentModels.prep_training(device=device.type)
    next_mfacs = agentModels.next_action_mean(sample)
    for _ in range(args.generator_epochs):
        agentModels.update_critic(sample,next_mfacs)
        agentModels.update_policies(sample)
        agentModels.update_all_targets()
    agentModels.prep_rollouts(device='cpu')


def main_loop():
    time_list = list()
    reward_list = list()
    label_list = list()
    rMean_list = list()
    rStd_list = list()
    for e_iter in range(args.epochs):
        rList = list()
        stdList = list()
        agentModels, discrimListBayesian, agentsInteract = create_networks()
        print(f"number of agents: {numAgents}\npolicy network: {agentModels.policies[0]}")
        print(f"number of agents: {numAgents}\ndiscriminator network: {discrimListBayesian[0].discrimNets[0]}")

        
        # train expert policy
        for i_iter in range(args.max_iter_num+1):
            """generate multiple trajectories that reach the minimum batch_size"""
            _, sample, log = agentsInteract.collect_samples(args.min_batch_size, cuda)
            t0 = time.time()
            update_params(sample, agentModels, discrimListBayesian)
            t1 = time.time()
            if i_iter % args.log_interval == 0:
                r_mean = np.mean(log['avg_reward'])
                r_std = np.std(log['avg_reward'])
                print('{}\tT_sample {:.2f}\tT_update {:.2f}\tR_avg {:.2f} +- {:.2f}, {:.2f}, {:.2f}, {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}'.format(
                    i_iter, log['sample_time'], t1 - t0, r_mean, r_std, log['avg_reward'][0], log['avg_reward'][1], log['avg_reward'][2],\
                    log['num_episodes'], log['num_steps']))
            reward_list.append(log['avg_reward'])
            time_list.append(i_iter)
            label_list.append(args.description[0])
            rList.append(r_mean)
            stdList.append(r_std)
            # save checkpoint
            if args.save_checkpoint_interval > 0 and i_iter % args.save_checkpoint_interval == 0:
                checkpoint_path_epoch = args.checkpoint_path + '_epoch' + str(e_iter) + '.tar'
                agentModels.save(checkpoint_path_epoch)
                agentModels.prep_rollouts(device='cpu')
                checkpointDictDiscrimNets = { 'discrimNets_'+str(di)+'_'+str(k) : v.state_dict() for di in range(args.discrim_net_nums) for k,v in enumerate(discrimListBayesian[di].discrimNets)}
                checkpointDictOptimizerDiscrims = { 'optimizerDiscrims_'+str(di)+'_'+str(k) : v.state_dict() for di in range(args.discrim_net_nums) for k,v in enumerate(discrimListBayesian[di].optimizerDiscrims)}
                checkpointDictAll = {}
                checkpointDictAll.update(checkpointDictDiscrimNets)
                checkpointDictAll.update(checkpointDictOptimizerDiscrims)
                checkpointD_path_epoch = args.checkpoint_path + 'D_epoch' + str(e_iter) + '.tar'
                torch.save(checkpointDictAll,checkpointD_path_epoch)
                df_vehicle = pd.DataFrame(env.xt_est)
                save_lane_vehicles_path_epoch = args.save_lane_vehicles_path + '_epoch' + str(e_iter) + '.pkl'
                df_vehicle.to_pickle(save_lane_vehicles_path_epoch)
                df_vehicle = pd.DataFrame(env.st_est)
                save_vehicle_location_path_epoch = args.save_vehicle_location_path + '_epoch' + str(e_iter) + '.pkl'
                df_vehicle.to_pickle(save_vehicle_location_path_epoch)
            """clean up gpu memory"""
            torch.cuda.empty_cache()

        
        # save training epoch
        rLarge = np.array(rList[-10:])
        rMean = np.mean(rLarge[np.argsort(rLarge)[-7:]])
        stdSmall = np.array(stdList[-10:])
        rStd = np.mean(stdSmall[np.argsort(stdSmall)[:7]])
        rMean_list.append(rMean)
        rStd_list.append(rStd)
        print('rMean {:.2f}\trStd {:.2f}'.format(rMean,rStd))
        data_dic = {'time': time_list, 'reward': reward_list, 'Algorithms': label_list}
        df = pd.DataFrame(data_dic)
        df.to_pickle(args.save_data_path)
    print('Epochs rMean {:.2f}\trStd {:.2f}'.format(np.mean(rMean_list),np.mean(rStd_list)))


main_loop()
