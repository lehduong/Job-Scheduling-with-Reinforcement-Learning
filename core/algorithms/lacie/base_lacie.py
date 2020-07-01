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
    INPUT_SEQ_DIM = 2  # hard code for load balance env
    CPC_HIDDEN_DIM = 32

    def __init__(self,
                 actor_critic,
                 lr,
                 value_coef,
                 entropy_coef,
                 state_to_input_seq=None,
                 expert=None,
                 il_coef=1):
        super().__init__(actor_critic, lr, value_coef, entropy_coef, expert, il_coef)
        self.state_to_input_seq = state_to_input_seq

        self.device = next(self.actor_critic.parameters()).device

        # create linear function for each time step
        self.advantage_encoder = nn.Sequential(
            nn.Linear(1, self.CPC_HIDDEN_DIM//2, bias=True),
            nn.ReLU(),
            nn.Linear(self.CPC_HIDDEN_DIM//2, self.CPC_HIDDEN_DIM, bias=True)
        ).to(self.device)

        # input sequence encoder
        self.input_seq_encoder = nn.GRU(
            self.INPUT_SEQ_DIM, self.CPC_HIDDEN_DIM, 1).to(self.device)

        self.cpc_optimizer = optim.Adam(chain(
            self.advantage_encoder.parameters(), self.input_seq_encoder.parameters()), lr=lr)

        self.softmax = nn.Softmax(dim=0)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def compute_contrastive_loss(self, rollouts, advantages):
        """
            Contrastive Predictive Coding for learning representation and density ratio
            :param rollouts: Storage's instance
            :param advantage: tensor of shape: (timestep, n_processes, 1)
        """
        # FIXME: only compatible with 1D observation
        num_steps, n_processes, _ = advantages.shape

        # INPUT SEQUENCES
        # the stochastic input will be defined by last 2 scalar
        input_seq = rollouts.obs[:-1, :, -2:]
        # reverse the input seq order since we want to compute from right to left
        input_seq = torch.flip(input_seq, [0])
        # encode the input sequence
        # n_steps x n_processes x hidden_dim
        input_seq, _ = self.input_seq_encoder(input_seq)
        # reverse back
        input_seq = torch.flip(input_seq, [0])

        # ADVANTAGES
        # encode
        # n_steps x hidden_dim x n_process
        advantages = self.advantage_encoder(
            advantages.reshape(-1, 1)).reshape(num_steps, n_processes, -1).permute(0, 2, 1)

        # compute nce
        contrastive_loss = 0
        correct = 0
        for i in range(num_steps):
            density_ratio = torch.mm(input_seq[i], advantages[i])
            # accuracy
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(
                density_ratio), dim=1), torch.arange(0, n_processes).to(self.device)))
            # nce
            contrastive_loss += torch.sum(
                torch.diag(self.log_softmax(density_ratio)))

        # log loss
        contrastive_loss /= -1*n_processes*num_steps
        # accuracy
        accuracy = 1.*correct.item()/(n_processes*num_steps)

        return contrastive_loss, accuracy

    def compute_weighted_advantages(self, rollouts, advantages):
        """
            Compute return for rollout experience with trained contrastive module
        """
        with torch.no_grad():
            # FIXME: only compatible with 1D observation
            num_steps, n_processes, _ = advantages.shape

            # INPUT SEQUENCES
            # the stochastic input will be defined by last 2 scalar
            input_seq = rollouts.obs[:-1, :, -2:]
            # reverse the input seq order since we want to compute from right to left
            input_seq = torch.flip(input_seq, [0])
            # encode the input sequence
            # output shape: n_steps x n_processes x hidden_dim
            input_seq, _ = self.input_seq_encoder(input_seq)
            # reverse back
            input_seq = torch.flip(input_seq, [0])

            # ADVANTAGES
            # encode
            # output shape: n_steps x hidden_dim x n_process
            encoded_advantages = self.advantage_encoder(
                advantages.reshape(-1, 1)).reshape(num_steps, n_processes, -1).permute(0, 2, 1)

            # weight of each advantage score
            weights = torch.zeros((num_steps, n_processes, 1)).to(
                self.device)

            for i in range(num_steps):
                # n_steps x n_steps
                density_ratio = self.softmax(
                    torch.mm(input_seq[i], encoded_advantages[i]))
                # take the diag element
                density_ratio = density_ratio.diag().reshape(n_processes, 1)

                weights[i] = density_ratio

            weights *= n_processes
            weights = 1/weights

        return advantages*weights
