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
                 regularize_coef,
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
                         regularize_coef=regularize_coef,
                         state_to_input_seq=state_to_input_seq,
                         expert=expert,
                         il_coef=il_coef,
                         num_cpc_steps=num_cpc_steps)
        self.max_grad_norm = max_grad_norm

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
        returns = rollouts.returns[:-1]

        # Value loss for updating Critic Net
        value_loss = advantages.pow(2).mean()

        # LEARNING CONTRASTIVE PREDICTIVE MODEL
        # compute contrastive loss and accuracy
        contrastive_loss, contrastive_accuracy, regularize_loss = self.compute_contrastive_loss(
            rollouts.obs, rollouts.actions, rollouts.masks, returns)
        contrastive_loss = contrastive_loss.item()
        regularize_loss = regularize_loss.item()
        # computed weighted advantage according to its dependency with input sequences

        # learn cpc model for n steps
        for _ in range(self.num_cpc_steps):
            cpc_loss, _, cpc_regularize_loss = self.compute_contrastive_loss(
                rollouts.obs, rollouts.actions, rollouts.masks, returns)

            self.cpc_optimizer.zero_grad()
            (cpc_loss + self.regularize_coef * cpc_regularize_loss).backward()

            # nn.utils.clip_grad_norm_(chain(self.advantage_encoder.parameters(),
            #                                self.input_seq_encoder.parameters(),
            #                                self.state_encoder.parameters(),
            #                                self.condition_encoder.parameters(),
            #                                self.action_encoder.parameters()),
            #                          self.max_grad_norm)

            self.cpc_optimizer.step()

        # IMPORTANCE: we need to compute the weighted before learn cpc model
        # FIXME: Move to training to top to verify if the model can estimate density ratio
        weighted_advantages = self.compute_weighted_advantages(
            rollouts.obs, rollouts.actions, rollouts.masks, returns) - values

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
        self.after_update()

        return {
            'value loss': value_loss.item(),
            'action loss': action_loss.item(),
            'entropy loss': dist_entropy.item(),
            'imitation loss': imitation_loss.item(),
            'imitation accuracy': imitation_accuracy,
            'contrastive loss': contrastive_loss,
            'contrastive accuracy': contrastive_accuracy,
            'regularize loss': regularize_loss
        }


class LACIE_A2C_Memory(LACIE_A2C):
    def __init__(self,
                 actor_critic,
                 value_coef,
                 entropy_coef,
                 regularize_coef,
                 eps=None,
                 alpha=None,
                 state_to_input_seq=None,
                 lr=1e-3,
                 max_grad_norm=None,
                 expert=None,
                 il_coef=1,
                 num_cpc_steps=10,
                 lacie_batch_size=64,
                 lacie_buffer=None,
                 use_memory_to_pred_weights=False):
        super().__init__(actor_critic,
                         value_coef,
                         entropy_coef,
                         regularize_coef,
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
        self.use_memory_to_pred_weights = use_memory_to_pred_weights

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
        returns = rollouts.returns[:-1]

        # Value loss for updating Critic Net
        value_loss = advantages.pow(2).mean()

        # LEARNING CONTRASTIVE PREDICTIVE MODEL
        # update LACIE_Storage
        self.lacie_buffer.insert(rollouts, returns)
        # compute contrastive loss and accuracy
        contrastive_loss, contrastive_accuracy, regularize_loss = self.compute_contrastive_loss(
            rollouts.obs, rollouts.actions, rollouts.masks, returns)
        contrastive_loss = contrastive_loss.item()
        regularize_loss = regularize_loss.item()

        # computed weighted advantage according to its dependency with input sequences
        # learn cpc model for n steps
        for _ in range(self.num_cpc_steps):
            data = self.lacie_buffer.sample()
            obs, actions, masks, advantages = data['obs'], data['actions'], data['masks'], data['advantages']
            cpc_loss, _, cpc_regularize_loss = self.compute_contrastive_loss(
                obs, actions, masks, advantages)

            self.cpc_optimizer.zero_grad()
            (cpc_loss + self.regularize_coef * cpc_regularize_loss).backward()

            # nn.utils.clip_grad_norm_(chain(self.advantage_encoder.parameters(),
            #                                self.input_seq_encoder.parameters(),
            #                                self.state_encoder.parameters(),
            #                                self.condition_encoder.parameters(),
            #                                self.action_encoder.parameters()),
            #                          self.max_grad_norm)

            self.cpc_optimizer.step()

        # IMPORTANCE: we need to compute the weighted before learn cpc model
        # FIXME: Move the cpc training on top to verify if it can learn useful estimation
        if not self.use_memory_to_pred_weights:
            weighted_advantages = self.compute_weighted_advantages(
                rollouts.obs, rollouts.actions, rollouts.masks, returns) - values
        else:
            data = self.lacie_buffer.sample_most_recent()
            obs, actions, masks, sample_advantages = data['obs'], data[
                'actions'], data['masks'], data['advantages']
            weighted_advantages = self.compute_weighted_advantages(
                obs, actions, masks, sample_advantages, rollouts.actions.shape[1]) - values

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
        self.after_update()

        return {
            'value loss': value_loss.item(),
            'action loss': action_loss.item(),
            'entropy loss': dist_entropy.item(),
            'imitation loss': imitation_loss.item(),
            'imitation accuracy': imitation_accuracy,
            'contrastive loss': contrastive_loss,
            'contrastive accuracy': contrastive_accuracy,
            'regularize loss': regularize_loss
        }
