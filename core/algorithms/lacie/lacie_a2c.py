import torch
import torch.nn as nn

from .base_lacie import LacieAlgo


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
                 state_to_input_seq=None,
                 lr=1e-3,
                 max_grad_norm=None,
                 expert=None,
                 il_coef=1):
        super().__init__(actor_critic=actor_critic,
                         lr=lr,
                         value_coef=value_coef,
                         entropy_coef=entropy_coef,
                         state_to_input_seq=state_to_input_seq,
                         expert=expert,
                         il_coef=il_coef)
        self.max_grad_norm = max_grad_norm

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[:-1].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        # computed weighted advantage according to its dependency with input sequences
        weighted_advantages = self.compute_weighted_advantages(
            rollouts, advantages)
        action_loss = -(weighted_advantages.detach() * action_log_probs).mean()

        # contrastive learning density ratio
        contrastive_loss, contrastive_accuracy = self.compute_contrastive_loss(
            rollouts, advantages)

        # imitation learning
        imitation_loss, imitation_accuracy = torch.tensor(0), 0
        if self.expert:
            imitation_loss, imitation_accuracy = self.imitation_learning(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(
                    -1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                self.expert)

        self.optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()

        (imitation_loss * self.il_coef + value_loss * self.value_coef + action_loss -
         dist_entropy * self.entropy_coef + contrastive_loss).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()
        self.cpc_optimizer.step()
        self.il_coef *= self.IL_DECAY_RATE

        return {
            'value loss': value_loss.item(),
            'action loss': action_loss.item(),
            'entropy loss': dist_entropy.item(),
            'imitation loss': imitation_loss.item(),
            'imitation accuracy': imitation_accuracy,
            'contrastive loss': contrastive_loss.item(),
            'contrastive accuracy': contrastive_accuracy
        }
