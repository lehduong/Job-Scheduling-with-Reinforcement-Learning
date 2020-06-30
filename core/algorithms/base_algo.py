from abc import ABC
from torch import optim, nn

import torch


class BaseAlgo(ABC):
    IL_DECAY_RATE = 0.995  # decay factor of imitation learning

    def __init__(self,
                 actor_critic,
                 lr,
                 value_coef,
                 entropy_coef,
                 expert=None,
                 il_coef=1):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr)

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.il_coef = il_coef
        self.expert = expert
        self.il_criterion = nn.CrossEntropyLoss()

    def update(self, rollouts):
        pass

    def imitation_learning(self, inputs, rnn_hxs, masks, expert):
        """
            Imitation learning loss

        :param inputs: state observations

        :param rnn_hxs: rnn hidden state

        :param masks: mask the final state with 0 value

        :param expert: a trained or heuristic agent

        :return: log probability of expert's actions
        """
        _, actor_features, _ = self.actor_critic.base(inputs, rnn_hxs, masks)
        dist = self.actor_critic.dist(actor_features)

        expert_actions = expert.act(inputs)

        il_loss = self.il_criterion(dist.probs, expert_actions.reshape(-1))
        accuracy = (torch.argmax(dist.probs, dim=1) ==
                    expert_actions.reshape(-1)).float().sum()/expert_actions.shape[0]

        return il_loss, accuracy
