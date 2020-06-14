import copy
from itertools import chain

import torch
from torch.nn import L1Loss
from torch.optim import Adam

from .pg import Policy


class MetaInputDependentPolicy(Policy):
    """
        Policy agent with meta critic. Support `train_and_predict_meta_critic` for learning to estimate value of new input\
                sequences
    """

    def __init__(self, obs_shape, action_shape, base=None, base_kwargs=None, num_inner_steps=1, adapt_lr=2e-3, adapt_criterion=L1Loss):
        super().__init__(obs_shape, action_shape, base, base_kwargs)
        self.num_inner_steps = num_inner_steps
        self.lr = adapt_lr
        self.adapt_criterion = adapt_criterion()

    @property
    def criterion(self):
        return self.adapt_criterion

    def train_and_predict_meta_critic(self, task_inputs, task_labels, meta_inputs, meta_labels):
        """
            Adapt the meta critic to new input-sequence and predict the values of new observation (with same input sequence)
                For simplicity, we adopt the **First-order MAML** https://arxiv.org/abs/1803.02999 \

            :param task_inputs: tuple of (obs, rnn_hxs, masks) - training inputs of roll out with same input sequence
            :param task_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values
            :param meta_inputs: tuple of (obs, rnn_hxs, masks) - testing inputs of roll out with same input sequence as training
            :param meta_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values
            :return: value prediction of meta_inputs and meta gradient of critic
        """
        # create new net and exclusively update this network
        fast_net = copy.deepcopy(self.base)
        task_optimizer = Adam(
            chain(fast_net.critic.parameters(), fast_net.gru.parameters()), lr=self.lr)

        task_obs, task_rnn_hxs, task_masks = task_inputs

        # Adapt to new task
        for step in range(self.num_inner_steps):
            task_preds, _, _ = fast_net(task_obs, task_rnn_hxs, task_masks)
            task_loss = self.adapt_criterion(task_preds, task_labels)

            # update the fast-adapted network
            task_optimizer.zero_grad()
            task_loss.backward()
            task_optimizer.step()

        # compute meta grad
        meta_obs, meta_rnn_hxs, meta_masks = meta_inputs
        meta_preds, _, _ = fast_net(meta_obs, meta_rnn_hxs, meta_masks)
        meta_loss = self.adapt_criterion(meta_preds, meta_labels)
        grads = torch.autograd.grad(meta_loss, chain(fast_net.critic.parameters(),
                                                     fast_net.gru.parameters()))
        meta_grads = {name: g for ((name, _), g) in zip(chain(fast_net.critic.named_parameters(),
                                                              fast_net.gru.named_parameters()),
                                                        grads)}

        return meta_preds, meta_grads

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

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
        action_log_probs = dist.log_probs(expert_actions)

        return action_log_probs

    def forward(self, inputs, rnn_hxs, masks):
        return self.base(inputs, rnn_hxs, masks)
