import torch

from torch import nn
from .pg import Policy


class ImitationLearner(Policy):
    """
        Actor-Critic agent which supports imitated learning scheme
    """

    def imitation_learning(self, inputs, rnn_hxs, masks, expert):
        """
            Imitation learning loss
        :param inputs: state observations
        :param rnn_hxs: rnn hidden state
        :param masks: mask the final state with 0 value
        :param expert: a trained or heuristic agent
        :return: log probability of expert's actions
        """
        _, actor_features, _ = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        expert_actions = expert.act(inputs)

        criterion = nn.CrossEntropyLoss()
        il_loss = criterion(dist.probs, expert_actions.reshape(-1))
        accuracy = (torch.argmax(dist.probs, dim=1) ==
                    expert_actions.reshape(-1)).float().sum()/expert_actions.shape[0]

        return il_loss, accuracy
