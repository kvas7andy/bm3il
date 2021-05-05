import torch
from utils import to_device

# estimate advantage from value
def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        # to be checked: why, td-lambda algorithm, which contains tau
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns

# estimate advantage from Q
def estimate_advantages_qvalue(rewards, masks, values, values_next, actions, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    advantages = torch.zeros_like(values)
    returns = values.clone()
    
    for i in reversed(range(rewards.size(0))):
        a_next = values_next[i].max(0)[1]
        advantages[i,actions[i]] = rewards[i] + gamma * values_next[i,a_next] * masks[i] - torch.mean(values[i,])
        returns[i,actions[i]] = rewards[i] + gamma * values_next[i,a_next] * masks[i]

    #advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


