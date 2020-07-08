"""
    Implement the algorithm `Learning to Assign Credit in Input-Driven Environment'
"""
from core.algorithms.base_algo import BaseAlgo
from torch import optim
from itertools import chain

import torch
import torch.nn as nn


class LacieAlgo(BaseAlgo):
    """
        Base class for LACIE algorithm. Support `cpc` (contrastive predictive coding) to estimate \
            the independent between input-process and future states.
            :param actor_critic: nn.Module - the actor critic object
            :param entropy_coef: float - weight of entropy loss
            :param max_grad_norm: float - maximum value of gradient
            :param n_steps: int - n-steps advantage estimation with hindsight
            :param state_to_input_seq: function - a function object that decompose input-processes from states\
                    the signature of function should be: foo(states) where states is torch.Tensor of shape \
                    T x N_processes x Obs_shape
    """
    UPPER_BOUND_CLIP_THRESHOLD = 2
    LOWER_BOUND_CLIP_THRESHOLD = 1/100
    WEIGHT_CLIP_GROWTH_FACTOR = 1.001
    WEIGHT_CLIP_DECAY_FACTOR = 0.999
    INPUT_SEQ_DIM = 2  # hard code for load balance env
    CPC_HIDDEN_DIM = 96

    def __init__(self,
                 actor_critic,
                 lr,
                 value_coef,
                 entropy_coef,
                 state_to_input_seq=None,
                 expert=None,
                 il_coef=1,
                 num_cpc_steps=10):
        super().__init__(actor_critic, lr, value_coef, entropy_coef, expert, il_coef)
        self.state_to_input_seq = state_to_input_seq
        self.num_cpc_steps = num_cpc_steps

        self.device = next(self.actor_critic.parameters()).device

        # encoder for advantages
        self.advantage_encoder = nn.Sequential(
            nn.Linear(1, self.CPC_HIDDEN_DIM//3, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.CPC_HIDDEN_DIM//3,
                      self.CPC_HIDDEN_DIM//3, bias=True)
        ).to(self.device)

        # encoder for states
        # FIXME: hard code for 1D env
        self.state_encoder = nn.Sequential(
            nn.Linear(self.actor_critic.obs_shape[0],
                      self.CPC_HIDDEN_DIM//3, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.CPC_HIDDEN_DIM//3, self.CPC_HIDDEN_DIM//3)
        ).to(self.device)

        # encoder for action
        self.action_encoder = nn.Sequential(
            nn.Embedding(self.actor_critic.action_space.n,
                         self.CPC_HIDDEN_DIM//3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.CPC_HIDDEN_DIM//3, self.CPC_HIDDEN_DIM//3)
        ).to(self.device)

        # encoding conditions (i.e. advantages + states + actions)
        self.condition_encoder = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.CPC_HIDDEN_DIM, self.CPC_HIDDEN_DIM, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.CPC_HIDDEN_DIM, self.CPC_HIDDEN_DIM)
        ).to(self.device)

        # input sequence encoder
        self.input_seq_encoder = nn.GRU(
            self.INPUT_SEQ_DIM, self.CPC_HIDDEN_DIM, 1).to(self.device)

        # optimizer to learn the parameters for cpc loss
        self.cpc_optimizer = optim.Adam(
            chain(
                self.advantage_encoder.parameters(),
                self.input_seq_encoder.parameters(),
                self.state_encoder.parameters(),
                self.action_encoder.parameters(),
                self.condition_encoder.parameters()
            ),
            lr=lr
        )

        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.cpc_criterion = nn.CrossEntropyLoss()
        self.upper_bound_clip_threshold = 1
        self.lower_bound_clip_threshold = 1

        # Initialize weights
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.input_seq_encoder._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.input_seq_encoder.__getattr__(
                        p), mode='fan_out', nonlinearity='relu')

        self.condition_encoder.apply(_weights_init)
        self.state_encoder.apply(_weights_init)
        self.action_encoder.apply(_weights_init)
        self.advantage_encoder.apply(_weights_init)

    def _encode_input_sequences(self, obs, masks):
        num_steps, n_processes, _ = obs.shape
        # obs is tensor of shape (n_steps + 1, n_processes, obs_shape)
        num_steps -= 1
        # INPUT SEQUENCES AND MASKS
        # the stochastic input will be defined by last 2 scalar
        input_seq = obs[1:, :, -2:]
        masks = masks[1:].reshape(num_steps, n_processes)
        # reverse the input seq order since we want to compute from right to left
        input_seq = torch.flip(input_seq, [0])
        masks = torch.flip(masks, [0])
        # encode the input sequence
        # Let's figure out which steps in the sequence have a zero for any agent
        has_zeros = ((masks[1:-1] == 0.0)
                     .any(dim=-1)
                     .nonzero()
                     .squeeze()
                     .cpu())

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [-1] + has_zeros + [num_steps - 1]

        outputs = []

        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            output, hxs = self.input_seq_encoder(
                input_seq[start_idx + 1: end_idx + 1],
                hxs * masks[start_idx].view(1, -1, 1) if start_idx > -1 else None)

            outputs.append(output)

        # x is a (T, N, -1) tensor
        input_seq = torch.cat(outputs, dim=0)
        assert len(input_seq) == num_steps
        # reverse back
        input_seq = torch.flip(input_seq, [0])

        return input_seq

    def _encode_advantages(self, advantages):
        # FIXME: only compatible with 1D observation
        num_steps, n_processes, _ = advantages.shape
        # ADVANTAGES
        # encode
        # n_steps  x n_process x hidden_dim/2
        advantages = self.advantage_encoder(
            advantages.reshape(-1, 1)).reshape(num_steps, n_processes, -1)

        return advantages

    def _encode_states(self, obs):
        num_steps, n_processes, _ = obs.shape
        num_steps -= 1
        # STATES
        # encode
        # n_steps x n_process x hidden_dim/2
        states = obs[:-1]
        # FIXME: hard code for 1D env
        states_shape = states.shape[2:][0]
        states = self.state_encoder(
            states.reshape(-1, states_shape)).reshape(num_steps, n_processes, -1)

        return states

    def _encode_actions(self, actions):
        num_steps, n_processes, _ = actions.shape
        # ACTION
        # encode
        # n_steps x n_process x 1
        actions = self.action_encoder(
            actions.reshape(-1)).reshape(num_steps, n_processes, -1)

        return actions

    def _encode_conditions(self, conditions):
        num_steps, n_processes, hidden_dim = conditions.shape
        # ACTION
        # encode
        # n_steps x n_process x 1
        conditions = self.condition_encoder(
            conditions.reshape(-1, hidden_dim)).reshape(num_steps, n_processes, -1)

        return conditions

    def compute_contrastive_loss(self, obs, actions, masks, advantages):
        """
            Contrastive Predictive Coding for learning representation and density ratio
            :param rollouts: Storage's instance
            :param advantage: tensor of shape: (timestep, n_processes, 1)
        """
        # FIXME: only compatible with 1D observation
        num_steps, n_processes, _ = advantages.shape

        # encoded all the input
        encoded_input_seq = self._encode_input_sequences(obs, masks)
        encoded_advantages = self._encode_advantages(advantages)
        encoded_states = self._encode_states(obs)
        encoded_actions = self._encode_actions(actions)

        # condition = STATE + ADVANTAGE + ACTIONS
        conditions = torch.cat(
            [encoded_advantages, encoded_states, encoded_actions], dim=-1)
        conditions = self._encode_conditions(conditions)
        # reshape to n_steps x hidden_dim x n_processes
        encoded_input_seq = encoded_input_seq.permute(0, 2, 1)

        # compute nce
        contrastive_loss = 0
        correct = 0

        label = torch.arange(0, n_processes).to(self.device)

        for i in range(num_steps):
            # f(Z, s0, a0, R) WITHOUT exponential
            f_value = torch.mm(conditions[i], encoded_input_seq[i])
            # accuracy
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(
                f_value), dim=1), torch.arange(0, n_processes).to(self.device)))
            # nce
            # contrastive_loss += torch.sum(
            #    torch.diag(self.log_softmax(f_value)))
            contrastive_loss += self.cpc_criterion(
                f_value, label)

        # log loss
        contrastive_loss /= n_processes*num_steps
        # accuracy
        accuracy = 1.*correct.item()/(n_processes*num_steps)

        return contrastive_loss, accuracy

    def compute_weighted_advantages(self, obs, actions, masks, advantages, n_envs=None):
        """
            Compute return for rollout experience with trained contrastive module
        """
        with torch.no_grad():
            # FIXME: only compatible with 1D observation
            num_steps, batch_size, _ = advantages.shape

            input_seq = self._encode_input_sequences(
                obs, masks)
            encoded_advantages = self._encode_advantages(advantages)
            encoded_states = self._encode_states(obs)
            encoded_actions = self._encode_actions(actions)

            # condition = STATE + ADVANTAGE
            conditions = torch.cat(
                [encoded_advantages, encoded_states, encoded_actions], dim=-1)
            # reshape to n_steps x hidden_dim x n_processes
            input_seq = input_seq.permute(0, 2, 1)

            # weight of each advantage score
            weights = torch.zeros((num_steps, n_envs if n_envs else batch_size, 1)).to(
                self.device)

            for i in range(num_steps):
                # n_steps x n_steps
                density_ratio = self.softmax(
                    torch.mm(conditions[i], input_seq[i]))
                if n_envs:
                    # N is not None => used memory for predicting weights
                    density_ratio = density_ratio[:n_envs, :n_envs]
                # take the diag element
                density_ratio = density_ratio.diag().reshape(
                    n_envs if n_envs else batch_size, 1)

                weights[i] = density_ratio

            weights *= batch_size
            weights = 1/(weights+1e-5)
            weights = torch.clamp(
                weights,
                self.upper_bound_clip_threshold,
                self.lower_bound_clip_threshold
            )

        weighted_advantages = advantages[:, :n_envs] * \
            weights if n_envs else advantages*weights

        return weighted_advantages

    def update_weight_clip_threshold(self):
        self.upper_bound_clip_threshold = min(
            self.upper_bound_clip_threshold * self.WEIGHT_CLIP_GROWTH_FACTOR,
            self.UPPER_BOUND_CLIP_THRESHOLD
        )
        self.lower_bound_clip_threshold = max(
            self.lower_bound_clip_threshold * self.WEIGHT_CLIP_DECAY_FACTOR,
            self.LOWER_BOUND_CLIP_THRESHOLD
        )

    def after_update(self):
        super().after_update()
        self.update_weight_clip_threshold()
