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
proj_direc = '/data/akvasov/bm3il/'
sys.path.append(os.path.abspath(os.path.join(
    proj_direc+'buildILGymEnvsRT', '..')))
from utils import *
from core.maagentMAACGymEnvs import AgentsInteraction
from algorithms.attention_sac import AttentionSAC
from utils.make_env import make_env
from utils.env_wrappers import StandardEnv
from utils.replay_memory_MAACGymEnvs import Memory
from models.mlp_discriminator import Discriminator
from utils.attention_reward import AttentionReward
from models.mlp_discriminator import Discriminator
from utils.svgd import RBF, phi

class ARGS():
    def __init__(self):
        # hyper-parameters for MAAC
        self.pol_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.attend_heads = 4
        self.pi_lr = 3e-3 #0.001
        self.q_lr = 3e-3 #0.001
        # target network learn rate
        self.tau = 3e-3 #1e-3
        self.gamma = 0.99
        self.reward_scale = 100
        self.episode_length = 25
        # attention reward
        self.discrim_net_nums = 2
        self.discrim_hidden_dim = 32
        self.discrim_attend_heads = 2
        self.discrim_out_dim = 1
        self.d_lr = 1e-3
        # hyper-parameters for IL
        self.render = False
        self.log_interval = 1
        self.gpu_index = 0
        self.seed = 1
        self.env_name = 'multi_speaker_listener'
        self.expert_traj_path = proj_direc + "buildILGymEnvsRT/data/" + self.env_name + "/exp_traj2.pkl"
        # hyper-parameters for MAIL
        self.expert_traj_len = int(3e4) #1e4
        self.reset_memory_interval = 10
        self.min_batch_size = 800 #4000
        self.sample_size = 800
        self.num_threads = 8 #4
        self.epochs = 6
        self.discriminator_epochs = 5 #5
        self.generator_epochs = 10
        self.max_iter_num = 5000 #int(1e4) #6000
        self.load_checkpoint = False
        self.save_checkpoint_interval = 100
        # GMMIL
        self.sigma_list = [sigma / 1.0 for sigma in [1, 2, 4, 8, 16]]
        # save directories
        self.checkpoint_path = proj_direc + "buildILGymEnvsRT/data/" + self.env_name + "/checkpoint_svgdGAIL1_2"
        self.description = 'svgdGAIL'
        self.save_data_path = proj_direc+"buildILGymEnvsRT/data/"+self.env_name+"_svgdGAIL1_2.pkl"
        
        
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
rawEnv = make_env(args.env_name, discrete_action=True)
env = StandardEnv(rawEnv)
numAgents = len(env.observation_space)
action_dims = [env.action_space[ai].n for ai in range(numAgents)]
observation_dims = [env.observation_space[ai].shape[0] for ai in range(numAgents)]
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
"""load expert trajectory"""
expert_traj = list()
for ai in range(numAgents):
    expert_traj_path_ai = args.expert_traj_path + '_agent_' + str(ai)
    expert_traj_ai = pd.read_pickle(expert_traj_path_ai).to_numpy()
    expert_traj_ai.dtype='float'
    expert_traj.append(expert_traj_ai[:args.expert_traj_len,:])

    
class DiscriminatorWrap:
    def __init__(self,numAgents):
        self.numAgents = numAgents
        self.discrimNets = []
        self.optimizerDiscrims = []
        
        
"""create agent (including actor and critic)"""
def create_networks():
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
        checkpoint_path_epoch = args.checkpoint_path + '_epoch' + str(0) + '.tar'
        agentModels = AttentionSAC.init_from_save(checkpoint_path_epoch, load_critic=True)
    agentModels.prep_rollouts(device='cpu')  
    discrimListBayesian = list()
    for _ in range(args.discrim_net_nums):
        discrimList = DiscriminatorWrap(numAgents)
        for ai in range(numAgents):  
            discrim_net = Discriminator(observation_dims[ai] + action_dims[ai], hidden_size=(args.discrim_hidden_dim,args.discrim_hidden_dim), activation='sigmoid')
            discrim_net.to(device)
            optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.d_lr)
            discrimList.discrimNets.append(discrim_net)
            discrimList.optimizerDiscrims.append(optimizer_discrim)
        discrimListBayesian.append(discrimList)
    # TBD
    if args.load_checkpoint:
        checkpointD_path_epoch = args.checkpoint_path + 'D_epoch' + str(0) + '.tar'
        checkpoint = torch.load(checkpointD_path_epoch)
        for k,v in checkpoint.items():
            kName,kD,kIdx = k.split('_')
            if kName=='discrimNets':
                discrimListBayesian[int(kD)].discrimNets[int(kIdx)].load_state_dict(v)
            elif kName=='optimizerDiscrims':
                discrimListBayesian[int(kD)].optimizerDiscrims[int(kIdx)].load_state_dict(v)      
    agentsInteract = AgentsInteraction(env, numAgents, agentModels, device, running_state=None, render=args.render, num_threads=args.num_threads)
    return agentModels, discrimListBayesian, agentsInteract
    
        
def update_params(batch, agentModels, discrimListBayesian, agentsInteract):
    """RL learn policy"""
    agentModels.prep_training(device=device.type)
    for _ in range(args.generator_epochs):
        sample = agentsInteract.batch2TensorSample(batch, cuda, sample_size=args.sample_size)
        """IRL estimate reward"""
        #obs, acs, rews, next_obs, dones = sample
        for ai in range(numAgents):
            rewards = list()
            for di in range(args.discrim_net_nums):
                dis_input_fake = torch.cat((sample[0][ai],sample[1][ai]),dim=1)
                g_o = discrimListBayesian[di].discrimNets[ai](dis_input_fake)
                rewards.append(-torch.log(g_o).to(dtype).to(device).detach()/args.discrim_net_nums)
            # use this reward -> il, comment thie line -> rl
            sample[2][ai] = torch.stack(rewards, dim=0).sum(dim=0)
        agentModels.update_critic(sample)
        agentModels.update_policies(sample)
        agentModels.update_all_targets()
    agentModels.prep_rollouts(device='cpu')
    
    
    """IRL update discriminator"""
    dataSize = sample[0][0].shape[0]
    for _ in range(args.discriminator_epochs):
        sample = agentsInteract.batch2TensorSample(batch, cuda, sample_size=args.sample_size, only_recent=True)
        exp_idx = random.sample(range(expert_traj[0].shape[0]), dataSize)
        for ai in range(numAgents):
            dis_input_fake = torch.cat((sample[0][ai],sample[1][ai]),dim=1)
            dis_input_real = torch.from_numpy(expert_traj[ai][exp_idx]).to(dtype).to(device)
            
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
        running_memory = [Memory() for _ in range(numAgents)]
        print(f"number of agents: {numAgents}\ndiscriminator network: {discrimListBayesian[0].discrimNets[0]}")

        
        # train expert policy
        for i_iter in range(args.max_iter_num):
            """generate multiple trajectories that reach the minimum batch_size"""
            batch, _, log = agentsInteract.collect_samples(args.min_batch_size, args.episode_length, cuda, running_memory=running_memory)
            t0 = time.time()        
            update_params(batch, agentModels, discrimListBayesian, agentsInteract)
            t1 = time.time()
            if (i_iter+1) % args.log_interval == 0:
                r_mean = np.mean(log['avg_reward'])
                r_std = np.std(log['avg_reward'])
                print('{}\tT_sample {:.2f}\tT_update {:.2f}\tR_avg {:.2f} +- {:.2f}, {:.2f}, {:.2f}, {:.2f}\tEpisodes {:.2f}\tSteps {:.2f}\t running_memory len {}'.format(
                    i_iter, log['sample_time'], t1 - t0, r_mean, r_std, log['avg_reward'][0], log['avg_reward'][1], log['avg_reward'][2],\
                    log['num_episodes'], log['num_steps'], len(running_memory[0])))
            reward_list.append(log['avg_reward'])
            time_list.append(i_iter)
            label_list.append(args.description[0])
            rList.append(r_mean)
            stdList.append(r_std)
            # clean running memory
            if (i_iter+1) % args.reset_memory_interval == 0:
                for ai in range(numAgents):
                    running_memory[ai].resetToLength(args.min_batch_size*(args.reset_memory_interval)//2)
            # save checkpoint
            if args.save_checkpoint_interval > 0 and (i_iter+1) % args.save_checkpoint_interval == 0:
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
