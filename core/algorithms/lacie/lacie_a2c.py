import torch
import torch.nn as nn

from itertools import chain
from torch import optim
from core.algorithms.lacie.base_lacie import LacieAlgo
from core.storage import LacieStorage


class LACIE_A2C(LacieAlgo):
    """
        Meta Input-dependent Baseline A2C. \
        This A2C class leverages input-dependent baseline, which is learned with meta learning, \
            to reduce variance when updating parameters
    """

    def __init__(self,
                 actor_critic,
                 value_coef,
                 entropy_coef,
                 eps=None,
                 alpha=None,
                 state_to_input_seq=None,
                 lr=1e-3,
                 max_grad_norm=None,
                 expert=None,
                 il_coef=1,
                 num_cpc_steps=10):
        super().__init__(actor_critic=actor_critic,
                         lr=lr,
                         value_coef=value_coef,
                         entropy_coef=entropy_coef,
                         state_to_input_seq=state_to_input_seq,
                         expert=expert,
                         il_coef=il_coef,
                         num_cpc_steps=num_cpc_steps)
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # Estimate baseline
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[:-1].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        # Value loss for updating Critic Net
        value_loss = advantages.pow(2).mean()

        # LEARNING CONTRASTIVE PREDICTIVE MODEL
        # compute contrastive loss and accuracy
        contrastive_loss, contrastive_accuracy = self.compute_contrastive_loss(
            rollouts.obs, rollouts.actions, rollouts.masks, advantages.detach())
        contrastive_loss = contrastive_loss.item()
        # computed weighted advantage according to its dependency with input sequences
        # IMPORTANCE: we need to compute the weighted before learn cpc model
        weighted_advantages = self.compute_weighted_advantages(
            rollouts, advantages.detach())
        # learn cpc model for n steps
        for _ in range(self.num_cpc_steps):
            cpc_loss, _ = self.compute_contrastive_loss(
                rollouts.obs, rollouts.actions, rollouts.masks, advantages.detach())

            self.cpc_optimizer.zero_grad()
            cpc_loss.backward()

            nn.utils.clip_grad_norm_(chain(self.advantage_encoder.parameters(),
                                           self.input_seq_encoder.parameters(),
                                           self.state_encoder.parameters(),
                                           self.action_encoder.parameters()),
                                     self.max_grad_norm)

            self.cpc_optimizer.step()

        # Action loss of Actor Net
        action_loss = -(weighted_advantages.detach() * action_log_probs).mean()

        # IMITATION LEARNING
        imitation_loss, imitation_accuracy = torch.tensor(
            0).to(rollouts.obs.device), 0
        if self.expert:
            imitation_loss, imitation_accuracy = self.imitation_learning(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                self.expert)

        self.optimizer.zero_grad()

        (imitation_loss * self.il_coef + value_loss * self.value_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()
        self.il_coef *= self.IL_DECAY_RATE

        return {
            'value loss': value_loss.item(),
            'action loss': action_loss.item(),
            'entropy loss': dist_entropy.item(),
            'imitation loss': imitation_loss.item(),
            'imitation accuracy': imitation_accuracy,
            'contrastive loss': contrastive_loss,
            'contrastive accuracy': contrastive_accuracy
        }


class LACIE_A2C_Memory(LACIE_A2C):
    def __init__(self,
                 actor_critic,
                 value_coef,
                 entropy_coef,
                 eps=None,
                 alpha=None,
                 state_to_input_seq=None,
                 lr=1e-3,
                 max_grad_norm=None,
                 expert=None,
                 il_coef=1,
                 num_cpc_steps=10,
                 lacie_batch_size=64,
                 lacie_buffer=None):
        super().__init__(actor_critic,
                         value_coef,
                         entropy_coef,
                         eps,
                         alpha,
                         state_to_input_seq,
                         lr,
                         max_grad_norm,
                         expert,
                         il_coef,
                         num_cpc_steps)
        self.lacie_batch_size = lacie_batch_size
        self.lacie_buffer = lacie_buffer

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # Estimate baseline
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[:-1].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        # Value loss for updating Critic Net
        value_loss = advantages.pow(2).mean()

        # LEARNING CONTRASTIVE PREDICTIVE MODEL
        # update LACIE_Storage
        self.lacie_buffer.insert(rollouts, advantages.detach())
        # compute contrastive loss and accuracy
        contrastive_loss, contrastive_accuracy = self.compute_contrastive_loss(
            rollouts.obs, rollouts.actions, rollouts.masks, advantages.detach())
        contrastive_loss = contrastive_loss.item()
        # computed weighted advantage according to its dependency with input sequences
        # IMPORTANCE: we need to compute the weighted before learn cpc model
        weighted_advantages = self.compute_weighted_advantages(
            rollouts, advantages.detach())
        # learn cpc model for n steps
        for _ in range(self.num_cpc_steps):
            data = self.lacie_buffer.sample(self.lacie_batch_size)
            obs, actions, masks, advantages = data['obs'], data['actions'], data['masks'], data['advantages']
            cpc_loss, _ = self.compute_contrastive_loss(
                obs, actions, masks, advantages)

            self.cpc_optimizer.zero_grad()
            cpc_loss.backward()

            nn.utils.clip_grad_norm_(chain(self.advantage_encoder.parameters(),
                                           self.input_seq_encoder.parameters(),
                                           self.state_encoder.parameters(),
                                           self.action_encoder.parameters()),
                                     self.max_grad_norm)

            self.cpc_optimizer.step()

        # Action loss of Actor Net
        action_loss = -(weighted_advantages.detach() * action_log_probs).mean()

        # IMITATION LEARNING
        imitation_loss, imitation_accuracy = torch.tensor(
            0).to(rollouts.obs.device), 0
        if self.expert:
            imitation_loss, imitation_accuracy = self.imitation_learning(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                self.expert)

        self.optimizer.zero_grad()

        (imitation_loss * self.il_coef + value_loss * self.value_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()
        self.il_coef *= self.IL_DECAY_RATE

        return {
            'value loss': value_loss.item(),
            'action loss': action_loss.item(),
            'entropy loss': dist_entropy.item(),
            'imitation loss': imitation_loss.item(),
            'imitation accuracy': imitation_accuracy,
            'contrastive loss': contrastive_loss,
            'contrastive accuracy': contrastive_accuracy
        }
