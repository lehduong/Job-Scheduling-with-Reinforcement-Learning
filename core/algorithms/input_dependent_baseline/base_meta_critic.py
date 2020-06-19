import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import L1Loss, MSELoss
from itertools import chain


class ActorMetaCriticAlgo:
    """
        Base class for algorithm (A2C, PPO, etc) which supports adapt meta critic to new \
            input sequences
    """

    def __init__(self,
                 actor_critic,
                 lr=7e-4,
                 adapt_lr=1e-3,
                 num_inner_steps=5,
                 adapt_criterion=MSELoss):
        self.actor_critic = actor_critic
        self.lr = lr
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        # meta critic args
        self.adapt_lr = adapt_lr
        self.num_inner_steps = num_inner_steps
        self.adapt_criterion = adapt_criterion()

        # imitation learning
        self.il_criterion = nn.CrossEntropyLoss()

    def adapt_and_predict(self, task_inputs, task_labels, meta_inputs, meta_labels):
        """
            Adapt the meta critic to new input-sequence and predict the values of new observation \
                (with same input sequence)
            For simplicity, we adopt the **First-order MAML** as in https://arxiv.org/abs/1803.02999 \

            :param task_inputs: tuple of (obs, rnn_hxs, masks) - training inputs of roll out with same input sequence

            :param task_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values

            :param meta_inputs: tuple of (obs, rnn_hxs, masks) - testing inputs of roll out with same input sequence as training

            :param meta_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values

            :return: value prediction of meta_inputs and meta gradient of critic
        """
        # create new net and exclusively update this network
        fast_net = copy.deepcopy(self.actor_critic.base)
        task_optimizer = optim.Adam(
            fast_net.parameters(), lr=self.lr)

        task_obs, task_rnn_hxs, task_masks = task_inputs

        # Adapt to new task
        for _ in range(self.num_inner_steps):
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
        grads = torch.autograd.grad(
            meta_loss, fast_net.parameters(), allow_unused=True)

        # create dictionary contains gradient of meta critic
        meta_grads = {name: g if g is not None else torch.zeros_like(weight)
                      for ((name, weight), g)
                      in zip(fast_net.named_parameters(),
                             grads)}

        return meta_preds, meta_grads

    def train_meta_critic_and_predict_values(self, rollouts):
        """
            Train the meta critic with rollout experience and return the predicted values \
            The adapted algorithm is described in Algorithm 1 of the paper \
                    https://arxiv.org/abs/1807.02264. We split the rollout into 2 half, \
                    the critic parameters adapting to the first half will give the prediction \
                    for second half. The critic parameters adapting to the second half will \
                    give the prediction for first half.

            :param rollouts: RolloutStorage's instance

            :return: input-dependent values
        """
        obs_shape = rollouts.obs.size()[2:]
        _, num_processes, _ = rollouts.rewards.size()

        # prepare input and output of meta learner
        # ie splitting them into 2
        task_pt = int(num_processes/2)
        # first half rollouts
        # num_steps * num_processes * input_shape
        first_obs = rollouts.obs[:-1, :task_pt, ...].reshape(-1, *obs_shape)
        first_rnn_hxs = rollouts.recurrent_hidden_states[0, :task_pt].reshape(
            -1, self.actor_critic.recurrent_hidden_state_size)
        first_mask = rollouts.masks[:-1, :task_pt].reshape(-1, 1)
        first_inputs = (first_obs, first_rnn_hxs, first_mask)
        first_labels = rollouts.returns[:-1, :task_pt, ...].reshape(-1, 1)
        # second half rollouts
        # num_steps * num_processes * input_shape
        second_obs = rollouts.obs[:-1, task_pt:, ...].reshape(-1, *obs_shape)
        second_rnn_hxs = rollouts.recurrent_hidden_states[0, task_pt:].reshape(
            -1, self.actor_critic.recurrent_hidden_state_size)
        second_mask = rollouts.masks[:-1, task_pt:].reshape(-1, 1)
        second_inputs = (second_obs, second_rnn_hxs, second_mask)
        second_labels = rollouts.returns[:-1, task_pt:, ...].reshape(-1, 1)

        # train meta network
        # the actor critic object must be instance of MetaCritic class
        second_values, second_meta_grads = self.adapt_and_predict(
            first_inputs, first_labels, second_inputs, second_labels)
        first_values, first_meta_grads = self.adapt_and_predict(
            second_inputs, second_labels, first_inputs, first_labels)
        values = torch.cat((first_values, second_values), dim=0)

        # update the meta critic
        self.update_meta_grads(
            [first_meta_grads, second_meta_grads], first_inputs, first_labels)

        # compute value loss
        value_loss = self.adapt_criterion(
            values, rollouts.returns[:-1].view(-1, 1))

        return values, value_loss.item()

    def update_meta_grads(self, grads, dummy_inputs, dummy_labels):
        """
            Set the gradient values from grads (dict) to actor_critic parameters and update 

            :param grads: list of OrderedDict - each element is the gradient from a task

            :param dummy_inputs: dummy inputs to activate the gradient of meta network

            :param dummy_labels: dummy labels to activate the gradient of meta network
        """
        keys = grads[0].keys()
        # multiple loss with value_loss_coef equivalent to multiple this coef with grad
        gradients = {k: sum(grad[k] for grad in grads) for k in keys}

        # compute dummy loss
        value_pred, _, _ = self.actor_critic.base(*dummy_inputs)
        loss = self.adapt_criterion(value_pred, dummy_labels)

        hooks = []
        for (k, v) in self.actor_critic.base.named_parameters():
            def get_closure():
                key = k

                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))

        # compute grad for curr step
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.actor_critic.base.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        for h in hooks:
            h.remove()

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
