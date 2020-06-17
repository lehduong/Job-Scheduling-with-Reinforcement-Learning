import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from core.algorithms.a2c_acktr import A2C_ACKTR
from core.agents.heuristic.load_balance import ShortestProcessingTimeAgent

DECAY_RATE = 0.999


class MetaInputDependentA2C(A2C_ACKTR):
    """
        Class support computing input-dependence baseline with meta learning
    """

    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=1e-3,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 expert=None,
                 il=10):
        super().__init__(actor_critic, value_loss_coef, entropy_coef,
                         lr, eps, alpha, max_grad_norm, acktr)
        self.critic_optimizer = optim.Adam(
            chain(actor_critic.base.critic.parameters(),
                  actor_critic.base.gru.parameters()),
            lr)
        self.actor_optimizer = optim.Adam(
            chain(actor_critic.base.actor.parameters(),
                  actor_critic.dist.parameters()),
            lr)

        del self.optimizer

        self.expert = expert
        self.il_coef = il

    def adapt_and_predict(self, rollouts):
        """
            Train the meta critic with rollout experience and return the predicted values

            The adapted algorithm is described in Algorithm 1 of the paper https://arxiv.org/abs/1807.02264 \
                    we split the rollout into 2 half, the critic parameters adapting to the first half will give 
                    the prediction for second half. The critic parameters adapting to the second half will give
                    the prediction for first half.
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
        second_value, second_meta_grads = self.actor_critic.train_and_predict_meta_critic(
            first_inputs, first_labels, second_inputs, second_labels)
        first_value, first_meta_grads = self.actor_critic.train_and_predict_meta_critic(
            second_inputs, second_labels, first_inputs, first_labels)
        values = torch.cat((first_value, second_value), dim=0)
        # update the meta critic
        self.update_meta_grads(
            [first_meta_grads, second_meta_grads], first_inputs, first_labels)
        # compute value loss
        criterion = self.actor_critic.criterion
        value_loss = criterion(values, rollouts.returns[:-1].view(-1, 1))

        return values, value_loss.item()

    def update_meta_grads(self, grads, dummy_inputs, dummy_labels):
        """
            Cummulate gradients across tasks 
            :param grads: list of OrderedDict - each element is the gradient from a task
            :param dummy_inputs: dummy inputs to activate the gradient of meta network
            :param dummy_labels: dummy labels to activate the gradient of meta network
        """
        keys = grads[0].keys()
        # multiple loss with value_loss_coef equivalent to multiple this coef with grad
        gradients = {k: sum(grad[k] for grad in grads) for k in keys}
        # compute dummy loss
        criterion = self.actor_critic.criterion
        value_pred, _, _ = self.actor_critic.base(*dummy_inputs)
        loss = criterion(value_pred, dummy_labels)

        hooks = []
        for (k, v) in chain(self.actor_critic.base.critic.named_parameters(), self.actor_critic.base.gru.named_parameters()):
            def get_closure():
                key = k

                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))

        # compute grad for curr step
        self.critic_optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.actor_critic.base.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        for h in hooks:
            h.remove()

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # imitation learning
        imitation_loss, accuracy = 0, 0
        if self.expert:
            imitation_loss, accuracy = self.actor_critic.imitation_learning(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                self.expert)
        # -----------------------------------------------------

        # action loss + entropy loss
        values, value_loss = self.adapt_and_predict(rollouts)
        _, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.actor_optimizer.zero_grad()

        # total loss
        loss = action_loss + self.il_coef * \
            imitation_loss - self.entropy_coef * dist_entropy
        loss.backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.actor_optimizer.step()

        # reduce the weight of imitation learning during training process
        self.il_coef = self.il_coef * DECAY_RATE

        return {
            'value loss': value_loss,
            'action loss': action_loss.item(),
            'entropy loss': dist_entropy.item(),
            'imitation loss': imitation_loss.item(),
            'accuracy': accuracy
        }
