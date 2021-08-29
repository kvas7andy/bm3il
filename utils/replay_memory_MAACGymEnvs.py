from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state',
                                       'dones'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch)) # Transition(state=(s_n1, s_n2,..., s_n{batch_size},
                                                                #action =  (a_n1, a_n2,..., a_n{batch_size}), ... )
        
    def reset(self):
        self.memory.clear()
        
    def resetToLength(self,length):
        self.memory = self.memory[len(self.memory)-length:]
        
    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
