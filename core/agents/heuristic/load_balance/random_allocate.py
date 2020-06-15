import torch

from .base import HeuristicAgent


class RandomAllocateAgent(HeuristicAgent):
    def act(self, states):
        num_env = states.shape[0]
        num_servers = states.shape[1] - 1

        return torch.randint(0, num_servers, (num_env, 1)).to(states.device)
