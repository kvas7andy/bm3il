import multiprocessing
import os
# import torch.multiprocessing as multiprocessing
# from multiprocessing import set_start_method
# set_start_method("spawn")
from torch import Tensor
from torch.autograd import Variable
from utils.replay_memory_MAACGymEnvs import Memory
from utils.torch import *
import numpy as np
import math
import time
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display
from torch.utils.tensorboard import SummaryWriter

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (env._spec.id, step, info))
    plt.axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    return None

def collect_samples(numAgents, episode_length, pid, queue, env, agentModels, custom_reward,
                    mean_action, render, running_state, min_batch_size):
    torch.randn(pid)
    log = dict()
    log['reward_list'] = list()
    memory = [Memory() for _ in range(numAgents)]
    num_steps = 0
    total_reward = [0 for _ in range(numAgents)]
    min_reward = [1e6 for _ in range(numAgents)]
    max_reward = [-1e6 for _ in range(numAgents)]
    total_c_reward = [0 for _ in range(numAgents)]
    min_c_reward = [1e6 for _ in range(numAgents)]
    max_c_reward = [-1e6 for _ in range(numAgents)]
    num_episodes = 0
    # tbd, should match build main dtype
    dtype = torch.float

    while num_steps < min_batch_size:
        state = env.reset()
        reward_episode = [0 for _ in range(numAgents)]

        if getattr(agentModels, "init_hidden", False):
            agentModels.init_hidden(batch_size=1)
        if getattr(agentModels, "init_latent", False):
            agentModels.init_latent(batch_size=1, first_time=True)

        # if not queue:
        #     agentModels.writer = SummaryWriter(os.path.join(agentModels.args.exper_path,
        #                                                      "logs/latent" + time.strftime("%Y-%m-%d_%H-%M-%S",
        #                                                                   time.localtime())))

        latents = []
        latents_infer = []
        for t in range(episode_length):
            # print('t:{:.1f}\tnum_steps:{:.1f}\tmin_batch_size:{:.1f}'.format(t,num_steps,min_batch_size))
            # tbd, add .to(dtype)
            torch_obs = [tensor(st).to(dtype).unsqueeze(0) for st in state]

            with torch.no_grad():
                if getattr(agentModels, "init_hidden", False):
                    torch_action = agentModels.step(torch_obs, t, explore=True)
                    latents.append(
                        agentModels.latent[:, 0, :agentModels.latent_dim].reshape(-1, agentModels.latent_dim))
                    latents_infer.append(
                        agentModels.latent_infer[:, 0, :agentModels.latent_dim].reshape(-1, agentModels.latent_dim))
                else:
                    torch_action = agentModels.step(torch_obs, explore=True)
                action = [ac.data.numpy()[0] for ac in torch_action]
            next_state, reward, done, _ = env.step(action)
            reward_episode = [sum(x) for x in zip(reward_episode, reward)]

            if custom_reward is not None:
                reward = custom_reward.expert_reward(state, action)
                total_c_reward = [sum(x) for x in zip(total_c_reward, reward)]
                min_c_reward = [min(x) for x in zip(min_c_reward, reward)]
                max_c_reward = [max(x) for x in zip(max_c_reward, reward)]

            for ai in range(numAgents):
                memory[ai].push(state[ai], action[ai], reward[ai], next_state[ai], done[ai])

            if render:
                show_state(env, step=t, info="")
                # env.render()
            # if done:
            #    break
            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward = [sum(x) for x in zip(total_reward, reward_episode)]
        min_reward = [min(x) for x in zip(min_reward, reward_episode)]
        max_reward = [max(x) for x in zip(max_reward, reward_episode)]
        log['reward_list'].append(reward_episode)

        if num_episodes + 1 == min_batch_size // episode_length and not queue \
                and getattr(agentModels, "args", False) and agentModels.args.writer:
            agentModels.args.writer.add_embedding(
                torch.cat([latent.unsqueeze(0) for latent in latents]).reshape(-1, agentModels.latent_dim),
                list(range(agentModels.args.n_agents)) * episode_length,
                global_step=agentModels.i_iter, tag="latent-cur")
            agentModels.args.writer.add_embedding(
                torch.cat([latent.unsqueeze(0) for latent in latents_infer]).reshape(-1, agentModels.latent_dim),
                list(range(agentModels.args.n_agents)) * episode_length,
                global_step=agentModels.i_iter, tag="latent-infer")

    # if not queue:
    #     agentModels.writer.flush()

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = [tr / num_episodes for tr in total_reward]  # tbd: num_episodes -> num_steps
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = [tr / num_steps for tr in total_c_reward]
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = [sum(y) for y in zip(*[x['total_reward'] for x in log_list])]
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = [tr / log['num_episodes'] for tr in log['total_reward']]
    log['max_reward'] = [max(y) for y in zip(*[x['max_reward'] for x in log_list])]
    log['min_reward'] = [min(y) for y in zip(*[x['min_reward'] for x in log_list])]
    log['reward_list'] = [y for x in log_list for y in x['reward_list']]
    # if 'total_c_reward' in log_list[0]:
    #    log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
    #    log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
    #    log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
    #    log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class AgentsInteraction:

    def __init__(self, env, numAgents, agentModels, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.numAgents = numAgents
        self.agentModels = agentModels
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def batch2TensorSample(self, batch, cuda, sample_size=None, only_recent=False, episode_l_sample=None):
        # if only recent == True, we only random shuffle the most recent sample_size
        if sample_size:
            if only_recent:
                if episode_l_sample:
                    ep_num = sample_size // episode_l_sample
                    if sample_size % episode_l_sample != 0: print(
                        'Warning: Not full sampling because sample_size is too small for full episode sample')
                    sampleIdx = np.arange(start=len(batch[0][0]) - ep_num * episode_l_sample, stop=len(batch[0][0]))
                    np.random.shuffle(sampleIdx.reshape(-1, episode_l_sample))

                else:
                    sampleIdx = np.arange(start=len(batch[0][0]) - sample_size, stop=len(batch[0][0]))
                    np.random.shuffle(sampleIdx)

            else:
                if episode_l_sample:
                    ep_num = sample_size // episode_l_sample
                    sampleIdx = np.random.choice(len(batch[0][0]) // episode_l_sample, size=ep_num, replace=False)
                    sampleIdx = (sampleIdx.reshape(-1, 1) + np.arange(episode_l_sample).reshape(1, episode_l_sample))
                else:
                    sampleIdx = np.random.choice(len(batch[0][0]), size=sample_size, replace=False)
        else:
            sampleIdx = np.arange(len(batch[0][0]))
        sampleIdx = sampleIdx.reshape(-1)
        if cuda:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        obs, acs, rews, next_obs, dones = [], [], [], [], []
        for sampleAi in batch:
            obsAi, acsAi, rewsAi, next_obsAi, donesAi = sampleAi
            obs.append(cast(obsAi)[sampleIdx, :])
            acs.append(cast(acsAi)[sampleIdx, :])
            rews.append(cast(rewsAi)[sampleIdx])
            next_obs.append(cast(next_obsAi)[sampleIdx, :])
            dones.append(cast(donesAi)[sampleIdx])
        return [obs, acs, rews, next_obs, dones]

    def collect_samples(self, min_batch_size, episode_length, cuda, running_memory=None):
        t_start = time.time()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads - 1):
            worker_args = (self.numAgents, episode_length, i + 1, queue, self.env, self.agentModels, self.custom_reward,
                           self.mean_action,
                           False, self.running_state, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(self.numAgents, episode_length, 0, None, self.env, self.agentModels,
                                      self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker in workers:
            worker.join()

        for worker_memory in worker_memories:
            for ai in range(self.numAgents):
                memory[ai].append(worker_memory[ai])
        if running_memory:
            for ai in range(self.numAgents):
                running_memory[ai].append(memory[ai])
            batch = [running_memory[ai].sample() for ai in range(self.numAgents)]
        else:
            batch = [memory[ai].sample() for ai in range(self.numAgents)]
        sample = self.batch2TensorSample(batch, cuda)

        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        return batch, sample, log
