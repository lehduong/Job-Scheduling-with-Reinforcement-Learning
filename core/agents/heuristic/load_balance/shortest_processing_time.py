import numpy as np
import torch

from .base import HeuristicAgent

j


class ShortestProcessingTimeAgent(HeuristicAgent):
    def __init__(self, service_rates=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]):
        self.service_rates = torch.tensor(service_rates).reshape(1, -1)

    def act(self, states):
        """
            Give actions for given states
            :param states: torch tensor of shape num_envs x (num_servers+1)
            :return: np.array of shape num_env x 1
        """
        processing_time = states[:, :-1]/self.service_rates.to(states.device)

        return torch.argmin(processing_time, dim=1, keepdims=True)
