from abc import ABC


class HeuristicAgent(ABC):
    def __init__(self):
        pass

    def act(self, states):
        """
            Give actions for given states
            :param states: torch tensor of shape num_envs x (num_servers+1)
        """
        pass
