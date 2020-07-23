import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import chain
from .base_lacie import LacieAlgo


class LACIE_PPO(LacieAlgo):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 regularize_coef,
                 state_to_input_seq=None,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 expert=None,
                 il_coef=1,
                 num_cpc_steps=10):
        super().__init__(actor_critic=actor_critic,
                         lr=lr,
                         value_coef=value_loss_coef,
                         entropy_coef=entropy_coef,
                         regularize_coef=regularize_coef,
                         state_to_input_seq=state_to_input_seq,
                         expert=expert,
                         il_coef=il_coef,
                         num_cpc_steps=num_cpc_steps)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        # contrastive learning loss
        contrastive_loss_epoch, contrastive_accuracy_epoch = self.compute_contrastive_loss(
            rollouts.obs, rollouts.actions, rollouts.masks, advantages.detach())
        contrastive_loss_epoch = contrastive_loss_epoch.item()

        # weighted advantages
        weighted_advantages = self.compute_weighted_advantages(
            rollouts.obs, rollouts.actions, rollouts.masks, advantages.detach())
        weighted_advantages = (weighted_advantages - weighted_advantages.mean()) / (
            weighted_advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        imitation_loss_epoch = 0
        accuracy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    weighted_advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    weighted_advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param,
                                                           self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # imitation learning
                imitation_loss, accuracy = torch.tensor(
                    0).to(action_loss.device), 0
                if self.expert:
                    imitation_loss, accuracy = self.imitation_learning(
                        rollouts.obs[:-1].view(-1, *obs_shape),
                        rollouts.recurrent_hidden_states[0].view(
                            -1, self.actor_critic.recurrent_hidden_state_size),
                        rollouts.masks[:-1].view(-1, 1),
                        self.expert)

                # contrastive learning density ratio
                contrastive_loss, _ = self.compute_contrastive_loss(
                    rollouts.obs, rollouts.actions, rollouts.masks, advantages)

                self.optimizer.zero_grad()
                self.cpc_optimizer.zero_grad()
                (imitation_loss * self.il_coef * self.value_coef + action_loss -
                 dist_entropy * self.entropy_coef + contrastive_loss).backward()
                nn.utils.clip_grad_norm_(chain(self.actor_critic.parameters(),
                                               self.input_seq_encoder.parameters(),
                                               self.advantage_encoder.parameters(),
                                               self.state_encoder.parameters(),
                                               self.condition_encoder.parameters(),
                                               self.action_encoder.parameters()),
                                         self.max_grad_norm)
                self.optimizer.step()
                self.cpc_optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                imitation_loss_epoch += imitation_loss.item()
                accuracy_epoch += accuracy

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        imitation_loss_epoch /= num_updates
        accuracy_epoch /= num_updates

        self.after_update()

        return {
            "value loss": value_loss_epoch,
            "action loss": action_loss_epoch,
            "entropy loss": dist_entropy_epoch,
            "imitation loss": imitation_loss_epoch,
            "accuracy": accuracy_epoch,
            "contrastive loss": contrastive_loss_epoch,
            "contrastive accuracy": contrastive_accuracy_epoch
        }


class LACIE_PPO_Memory(LACIE_PPO):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 regularize_coef,
                 state_to_input_seq=None,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 expert=None,
                 il_coef=1,
                 num_cpc_steps=10,
                 lacie_buffer=None,
                 lacie_batch_size=64,
                 use_memory_to_pred_weights=False):
        super().__init__(actor_critic,
                         clip_param,
                         ppo_epoch,
                         num_mini_batch,
                         value_loss_coef,
                         entropy_coef,
                         regularize_coef,
                         state_to_input_seq,
                         lr,
                         eps,
                         max_grad_norm,
                         use_clipped_value_loss,
                         expert,
                         il_coef,
                         num_cpc_steps)

        self.lacie_buffer = lacie_buffer
        self.lacie_buffer_size = lacie_batch_size
        self.use_memory_to_pred_weights = use_memory_to_pred_weights

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        # update LACIE_Storage
        self.lacie_buffer.insert(rollouts, advantages.detach())

        # contrastive learning loss
        contrastive_loss_epoch, contrastive_accuracy_epoch, _ = self.compute_contrastive_loss(
            rollouts.obs, rollouts.actions, rollouts.masks, advantages.detach())
        contrastive_loss_epoch = contrastive_loss_epoch.item()

        # ---------------------------------------------------------------------------
        # learn cpc model for n steps

        for _ in range(self.num_cpc_steps):
            data = self.lacie_buffer.sample()
            obs, actions, masks, sample_advantages = data['obs'], data['actions'], data['masks'], data['advantages']
            cpc_loss, _, cpc_regularize_loss = self.compute_contrastive_loss(
                obs, actions, masks, sample_advantages)

            self.cpc_optimizer.zero_grad()
            (cpc_loss + self.regularize_coef * cpc_regularize_loss).backward()

            nn.utils.clip_grad_norm_(chain(self.advantage_encoder.parameters(),
                                           self.input_seq_encoder.parameters(),
                                           self.state_encoder.parameters(),
                                           self.condition_encoder.parameters(),
                                           self.action_encoder.parameters()),
                                     self.max_grad_norm)

            self.cpc_optimizer.step()

        # weighted advantages
        if not self.use_memory_to_pred_weights:
            weighted_advantages = self.compute_weighted_advantages(
                rollouts.obs, rollouts.actions, rollouts.masks, advantages.detach())
        else:
            data = self.lacie_buffer.sample_most_recent()
            obs, actions, masks, sample_advantages = data['obs'], data[
                'actions'], data['masks'], data['advantages']
            weighted_advantages = self.compute_weighted_advantages(
                obs, actions, masks, sample_advantages, rollouts.actions.shape[1])
        # normalize advantages
        # TODO: Conduct Ablation Study to verify if we should normalize the advantages or not
        weighted_advantages = (weighted_advantages - weighted_advantages.mean()) / (
            weighted_advantages.std() + 1e-5)

        # ---------------------------------------------------------------------------
        # learn actor and critic

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        imitation_loss_epoch = 0
        accuracy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    weighted_advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    weighted_advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param,
                                                           self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # imitation learning
                imitation_loss, accuracy = torch.tensor(
                    0).to(action_loss.device), 0
                if self.expert:
                    imitation_loss, accuracy = self.imitation_learning(
                        rollouts.obs[:-1].view(-1, *obs_shape),
                        rollouts.recurrent_hidden_states[0].view(
                            -1, self.actor_critic.recurrent_hidden_state_size),
                        rollouts.masks[:-1].view(-1, 1),
                        self.expert)

                self.optimizer.zero_grad()
                (imitation_loss * self.il_coef * self.value_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                imitation_loss_epoch += imitation_loss.item()
                accuracy_epoch += accuracy

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        imitation_loss_epoch /= num_updates
        accuracy_epoch /= num_updates

        self.after_update()

        return {
            "value loss": value_loss_epoch,
            "action loss": action_loss_epoch,
            "entropy loss": dist_entropy_epoch,
            "imitation loss": imitation_loss_epoch,
            "accuracy": accuracy_epoch,
            "contrastive loss": contrastive_loss_epoch,
            "contrastive accuracy": contrastive_accuracy_epoch
        }
