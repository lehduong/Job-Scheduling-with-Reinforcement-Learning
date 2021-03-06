import numpy as np
import torch

from .base import HeuristicAgent


class LeastWorkAgent(HeuristicAgent):
    def act(self, states):
        """
            Give actions for given states
            :param states: torch tensor of shape num_envs x (num_servers+1)
            :return: np.array of shape num_env x 1
        """
        return torch.argmin(states[:, :-2], dim=1, keepdims=True).to(states.device)
