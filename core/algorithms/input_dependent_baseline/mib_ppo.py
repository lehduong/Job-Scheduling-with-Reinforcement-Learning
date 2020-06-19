import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.algorithms.input_dependent_baseline.base_meta_critic import ActorMetaCriticAlgo

DECAY_RATE = 0.995


class MIB_PPO(ActorMetaCriticAlgo):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 entropy_coef,
                 lr=None,
                 adapt_lr=None,
                 num_inner_steps=5,
                 max_grad_norm=None,
                 expert=None,
                 il=10):

        super().__init__(actor_critic, lr, adapt_lr, num_inner_steps)

        # PPO Args
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.expert = expert
        self.il_coef = il

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # action loss + entropy loss
        value_preds, value_loss = self.train_meta_critic_and_predict_values(
            rollouts)
        _, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        value_preds = value_preds.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - value_preds
        # lehduong: Implementation details: normalized the advantage values
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        advantages = advantages.detach()

        action_loss_epoch = 0
        dist_entropy_epoch = 0
        imitation_loss_epoch = 0
        accuracy_epoch = 0

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    _, _, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                value_preds, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # imitation learning
                imitation_loss, accuracy = torch.tensor(0), 0
                if self.expert:
                    imitation_loss, accuracy = self.imitation_learning(
                        rollouts.obs[:-1].view(-1, *obs_shape),
                        rollouts.recurrent_hidden_states[0].view(
                            -1, self.actor_critic.recurrent_hidden_state_size),
                        rollouts.masks[:-1].view(-1, 1),
                        self.expert)

                self.optimizer.zero_grad()
                (imitation_loss * self.il_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                imitation_loss_epoch += imitation_loss.item()
                accuracy_epoch += accuracy

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        imitation_loss_epoch /= num_updates
        accuracy_epoch /= num_updates

        self.il_coef *= DECAY_RATE

        return {
            "value loss": value_loss,
            "action loss": action_loss_epoch,
            "imitation loss": imitation_loss_epoch,
            "accuracy": accuracy_epoch,
            "entropy loss": dist_entropy_epoch
        }
