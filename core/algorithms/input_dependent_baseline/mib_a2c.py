import torch
import torch.nn as nn

from .base_meta_critic import ActorMetaCriticAlgo

DECAY_RATE = 0.995


class MIB_A2C(ActorMetaCriticAlgo):
    """
        Meta Input-dependent Baseline A2C. \
        This A2C class leverages input-dependent baseline, which is learned with meta learning, \
            to reduce variance when updating parameters
    """

    def __init__(self,
                 actor_critic,
                 entropy_coef,
                 lr=1e-3,
                 adapt_lr=1e-3,
                 num_inner_steps=5,
                 max_grad_norm=None,
                 expert=None,
                 il=10):
        super().__init__(actor_critic=actor_critic,
                         lr=lr,
                         adapt_lr=adapt_lr,
                         num_inner_steps=num_inner_steps)
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.expert = expert
        self.il_coef = il

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # action loss + entropy loss
        values, value_loss = self.train_meta_critic_and_predict_values(
            rollouts)
        _, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        # Normalize advantages?
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-5)

        action_loss = -(advantages.detach() * action_log_probs).mean()

        # imitation learning
        imitation_loss, accuracy = torch.tensor(0), 0
        if self.expert:
            imitation_loss, accuracy = self.imitation_learning(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                self.expert)
        # -----------------------------------------------------

        self.optimizer.zero_grad()

        # total loss
        loss = action_loss + self.il_coef * \
            imitation_loss - self.entropy_coef * dist_entropy
        loss.backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()

        # reduce the weight of imitation learning during training process
        self.il_coef = self.il_coef * DECAY_RATE

        return {
            'value loss': value_loss,
            'action loss': action_loss.item(),
            'entropy loss': dist_entropy.item(),
            'imitation loss': imitation_loss.item(),
            'accuracy': accuracy
        }
