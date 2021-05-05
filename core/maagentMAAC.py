import multiprocessing
from torch import Tensor
from torch.autograd import Variable
from utils.replay_memory_MAAC import Memory
from utils.torch import *
import math
import time

 
def collect_samples(numAgents, pid, queue, env, agentModels, custom_reward,
                    mean_action, render, running_state, min_batch_size, meanField, type2agent, agent2type):
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
    listSum = lambda x: np.array([sum(y) for y in zip(*x)]) # list of np array -> array of list element-wise sum
    if meanField:
        ntypes = len(type2agent)

    while num_steps < min_batch_size:
        state, valid_action_list = env.reset()
        if meanField:
            obsMF = [listSum([state[ai] for ai in type2agent[ti]]) for ti in range(ntypes)]
            state = [np.concatenate((state[ai],obsMF[agent2type[ai]])) for ai in range(numAgents)]
        reward_episode = [0 for _ in range(numAgents)]

        for t in range(10000):
            torch_obs = [tensor(st).to(dtype).unsqueeze(0) for st in state]
            torch_masks = [tensor(m).to(dtype).unsqueeze(0) for m in valid_action_list]
            
            with torch.no_grad():
                torch_action = agentModels.step(torch_obs, masks=torch_masks, explore=True)
                action = [ac.data.numpy()[0] for ac in torch_action]
            next_state, reward, valid_action_list, done, _ = env.step(action)
            if meanField:
                obsMF = [listSum([next_state[ai] for ai in type2agent[ti]]) for ti in range(ntypes)]
                next_state = [np.concatenate((next_state[ai],obsMF[agent2type[ai]])) for ai in range(numAgents)]
            reward_episode = [sum(x) for x in zip(reward_episode,reward)]
            
            if custom_reward is not None:
                reward = custom_reward.expert_reward(state, action)
                total_c_reward = [sum(x) for x in zip(total_c_reward, reward)]
                min_c_reward = [min(x) for x in zip(min_c_reward, reward)]
                max_c_reward = [max(x) for x in zip(max_c_reward, reward)]

            for ai in range(numAgents):
                memory[ai].push(state[ai], action[ai], valid_action_list[ai], reward[ai], next_state[ai], done)

            if render:
                env.render()
            if done:
                break
            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward = [sum(x) for x in zip(total_reward, reward_episode)]
        min_reward = [min(x) for x in zip(min_reward, reward_episode)]
        max_reward = [max(x) for x in zip(max_reward, reward_episode)]
        log['reward_list'].append(reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = [tr/num_episodes for tr in total_reward] # tbd: num_episodes -> num_steps
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = [tr/num_steps for tr in total_c_reward]
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
    log['avg_reward'] = [tr/log['num_episodes'] for tr in log['total_reward']]
    log['max_reward'] = [max(y) for y in zip(*[x['max_reward'] for x in log_list])]
    log['min_reward'] = [min(y) for y in zip(*[x['min_reward'] for x in log_list])]
    log['reward_list'] = [y for x in log_list for y in x['reward_list']]
    #if 'total_c_reward' in log_list[0]:
    #    log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
    #    log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
    #    log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
    #    log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log

class AgentsInteraction:

    def __init__(self, env, numAgents, agentModels, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1, meanField=False, type2agent=None, agent2type=None):
        self.env = env
        self.numAgents = numAgents
        self.agentModels = agentModels
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads
        self.meanField = meanField
        self.type2agent = type2agent
        self.agent2type = agent2type
    
    def batch2TensorSample(self,batch,cuda):
        if cuda:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        obs, acs, masks, rews, next_obs, dones = [], [], [], [], [], []
        for sampleAi in batch:
            obsAi, acsAi, masksAi, rewsAi, next_obsAi, donesAi = sampleAi
            obs.append(cast(obsAi))
            acs.append(cast(acsAi))
            masks.append(cast(masksAi))
            rews.append(cast(rewsAi))
            next_obs.append(cast(next_obsAi))
            dones.append(cast(donesAi))
        return([obs, acs, masks, rews, next_obs, dones])
    
    def collect_samples(self, min_batch_size, cuda):
        t_start = time.time()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (self.numAgents, i+1, queue, self.env, self.agentModels, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size, self.meanField, self.type2agent, self.agent2type)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(self.numAgents, 0, None, self.env, self.agentModels, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, self.meanField, self.type2agent, self.agent2type)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            for ai in range(self.numAgents):
                memory[ai].append(worker_memory[ai])
        batch = [memory[ai].sample() for ai in range(self.numAgents)]
        sample = self.batch2TensorSample(batch, cuda)
            
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        return batch, sample, log
