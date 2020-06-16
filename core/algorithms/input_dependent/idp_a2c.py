import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from core.algorithms.a2c_acktr import A2C_ACKTR
from core.agents.heuristic.load_balance import ShortestProcessingTimeAgent


class MetaInputDependentA2C(A2C_ACKTR):
    """
        Class support computing input-dependence baseline with meta learning
    """

    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 critic_lr=1e-3,
                 actor_lr=1e-3,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):
        super().__init__(actor_critic, value_loss_coef, entropy_coef,
                         actor_lr, eps, alpha, max_grad_norm, acktr)
        self.meta_optimizer = optim.Adam(
            chain(actor_critic.base.critic.parameters(),
                  actor_critic.base.gru.parameters()),
            critic_lr)
        self.optimizer = optim.Adam(
            chain(actor_critic.base.actor.parameters(),
                  actor_critic.dist.parameters()),
            actor_lr)

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
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

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
        self.meta_optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.actor_critic.base.critic.parameters(), self.max_grad_norm)
        self.meta_optimizer.step()

        for h in hooks:
            h.remove()

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, value_loss = self.adapt_and_predict(rollouts)

        # imitation learning
        expert = ShortestProcessingTimeAgent()
        imitation_loss = self.actor_critic.imitation_learning(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            expert)
        # -----------------------------------------------------

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

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (action_loss + imitation_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss, action_loss.item(), dist_entropy.item()
