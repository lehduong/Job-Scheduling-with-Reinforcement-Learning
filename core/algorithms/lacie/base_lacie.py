"""
    Implement the algorithm `Learning to Assign Credit in Input-Driven Environment'
"""
from abc import ABC

import torch
import torch.nn as nn


class LacieAlgo(ABC):
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

    def __init__(self,
                 actor_critic,
                 entropy_coef,
                 state_to_input_seq,
                 max_grad_norm=0.5,
                 n_steps=20):
        super().__init__()
        self.actor_critic = actor_critic
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.state_to_input_seq = state_to_input_seq

        # contrastive learning to learn the independent ratio
        hidden_dim = actor_critic.base.output_size
        device = actor_critic.device
        # create linear function for each time step
        self.wk_state = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for i in range(n_steps)]).to(device)

        self.softmax = nn.Softmax(dim=0)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def cpc(self, obs_feat, action_feat):
        """
            Contrastive Predictive Coding for learning representation and density ratio
            :param obs_feat: tensor of shape: (timestep, n_processes, hidden_size)
            :param action_feat: tensor of shape: (timestep, n_processes, hidden_size)
        """
        num_steps, n_processes, hidden_size = action_feat.shape

        # s_t to compute p(s_t|s_k)
        state_condition = obs_feat[0].view(n_processes, hidden_size)

        # compute W_i*c_t
        # num_steps * n_processes * hidden_size
        pred_state = torch.empty(
            num_steps, n_processes, hidden_size).float().to(self.actor_critic.device)
        for i in range(self.n_steps):
            # condition s_t
            linear_state = self.wk_state[i]
            pred_state[i] = linear_state(state_condition)

        # transpose pred_state and pred_state_action to num_steps, hidden_size, n_processes
        pred_state = pred_state.permute(0, 2, 1)

        # compute nce
        nce_state = 0
        correct_state = 0
        for i in range(self.n_steps):
            state_total = torch.mm(obs_feat[i], pred_state[i])
            # accuracy
            correct_state += torch.sum(torch.eq(torch.argmax(self.softmax(
                state_total), dim=0), torch.arange(0, n_processes).to(self.actor_critic.device)))
            # nce
            nce_state += torch.sum(torch.diag(self.log_softmax(state_total)))

        # log loss
        nce_state /= -1*n_processes*self.n_steps
        # accuracy
        accuracy_state = 1.*correct_state.item()/(n_processes*num_steps)

        return accuracy_state, nce_state

    def compute_return(self, rollouts):
        """
            Compute return for rollout experience with trained contrastive module
        """
        pass
