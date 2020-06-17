import torch

from .shortest_processing_time import ShortestProcessingTimeAgent


class EarliestCompletionTimeAgent(ShortestProcessingTimeAgent):
    def act(self, states):
        """
            Give actions for given states
            :param states: torch tensor of shape num_envs x (num_servers+1)
            :return: np.array of shape num_env x 1
        """
        processing_time = states[:, :-2] / self.service_rates.to(states.device)
        completion_time = states[:, :-2] + processing_time

        return torch.argmin(completion_time, dim=1, keepdims=True)
