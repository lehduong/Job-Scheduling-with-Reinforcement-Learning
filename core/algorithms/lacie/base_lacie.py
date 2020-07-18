"""
    Implement the algorithm `Learning to Assign Credit in Input-Driven Environment'
"""
from core.algorithms.base_algo import BaseAlgo
from torch import optim
from itertools import chain

import torch
import random
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
    UPPER_BOUND_CLIP_THRESHOLD = 4
    LOWER_BOUND_CLIP_THRESHOLD = 1/100
    WEIGHT_CLIP_GROWTH_FACTOR = 1.002
    WEIGHT_CLIP_DECAY_FACTOR = 0.998
    CPC_HIDDEN_DIM = 96
    POSITION_ENC_DIM = CPC_HIDDEN_DIM//3
    INPUT_ENC_DIM = 32

    def __init__(self,
                 actor_critic,
                 lr,
                 value_coef,
                 entropy_coef,
                 regularize_coef=0.05,
                 state_to_input_seq=None,
                 expert=None,
                 il_coef=1,
                 num_cpc_steps=10):
        super().__init__(actor_critic, lr, value_coef, entropy_coef, expert, il_coef)
        self.regularize_coef = regularize_coef
        self.state_to_input_seq = state_to_input_seq
        self.num_cpc_steps = num_cpc_steps

        self.device = next(self.actor_critic.parameters()).device

        # encoder for advantages
        self.advantage_encoder = nn.Sequential(
            nn.Linear(self.POSITION_ENC_DIM, self.CPC_HIDDEN_DIM//3, bias=True),
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
            self.INPUT_ENC_DIM, self.CPC_HIDDEN_DIM, 1).to(self.device)

        # optimizer to learn the parameters for cpc loss
        self.cpc_optimizer = optim.RMSprop(
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
        self.regularization_criterion = nn.L1Loss()

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

        # transform input_seq with fourier features
        jobs, intervals = input_seq[:, :, 0].reshape(-1, 1), input_seq[:, :, 1].reshape(-1, 1)
        jobs, intervals = self.encode_fourier_features(jobs, self.INPUT_ENC_DIM//2), self.encode_fourier_features(intervals, self.INPUT_ENC_DIM//2)
        jobs = jobs.reshape(num_steps, n_processes, self.INPUT_ENC_DIM//2)
        intervals = intervals.reshape(num_steps, n_processes, self.INPUT_ENC_DIM//2)
        input_seq = torch.cat([jobs, intervals], dim=-1)

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
        advantages = advantages.reshape(-1, 1)
        advantages = self.encode_fourier_features(advantages, self.POSITION_ENC_DIM)
        advantages = self.advantage_encoder(advantages).reshape(num_steps, n_processes, -1)

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
        # create label mask
        label = torch.tensor(torch.arange(
            0, n_processes).tolist() * num_steps).to(self.device)

        # broadcast compute matmul
        f_value = torch.bmm(
            conditions, encoded_input_seq).reshape(-1, n_processes)

        # compute accuracy
        correct = torch.sum(torch.eq(torch.argmax(
            self.softmax(f_value), dim=1), label))
        accuracy = correct.item()/(n_processes*num_steps)

        # compute loss
        contrastive_loss = self.cpc_criterion(f_value, label)
        regularization_loss = self.regularization_criterion(
            self.softmax(f_value) * n_processes, torch.ones_like(f_value))

        return contrastive_loss, accuracy, regularization_loss

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
                self.lower_bound_clip_threshold,
                self.upper_bound_clip_threshold
            )

        if random.randint(0, 9) == 0:
            print('weights mean: ', weights.mean())
            print('weights max: ', weights.max())
            print('weights min: ', weights.min())
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

    def encode_fourier_features(self, x, d=10):
        """
            Encode input with fourier features according to https://arxiv.org/abs/2006.10739
            :param x: torch.Tensor of shape Nx1
            :param d: int - encoded dimension
        """
        if (d//2)*2-d != 0:
            raise ValueError("Dimension must be even number...")
        N = x.shape[0]

        position_enc = torch.zeros(N, d).to(self.device)
        idx = torch.arange(d//2).reshape(1, -1).to(self.device)

        position_enc[:, 0::2] = torch.sin(x*2**idx)
        position_enc[:, 1::2] = torch.cos(x*2**idx)

        return position_enc
