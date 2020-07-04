import torch
import numpy as np


class LacieStorage(object):
    def __init__(self, num_steps, obs_shape, action_space,
                 max_size=10000,
                 batch_size=64,
                 n_processes=16):
        # obs
        self.obs = torch.zeros(max_size, num_steps + 1, * obs_shape)

        # action
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(max_size, num_steps, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        # mask
        self.masks = torch.ones(max_size, num_steps + 1, 1)

        # advantages
        self.advantages = torch.zeros(max_size, num_steps, 1)

        self.ptr, self.size, self.max_size = 0, 0, max_size

        self.batch_size = batch_size
        self.n_processes = n_processes

    def to(self, device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.advantages = self.advantages.to(device)

    def insert(self, rollouts, advantages):
        """
            Update the buffer with new rollouts from Storages mem
            :param obs: torch.Tensor of shape (num_steps + 1, n_processes, obs_shape)
            :param actions: torch.Tensor of shape (num_steps, n_processes, action_shape)
            :param masks: torch.Tensor of shape (num_steps + 1, n_processes, 1)
            :param advantages: torch.Tensor of shape (num_steps + 1, n_processes, 1)
        """
        obs = rollouts.obs.permute(1, 0, 2)
        actions = rollouts.actions.permute(1, 0, 2)
        masks = rollouts.masks.permute(1, 0, 2)
        advantages = advantages.permute(1, 0, 2)
        n = obs.shape[0]

        idxs = np.arange(self.ptr, self.ptr + n) % self.max_size
        self.obs[idxs].copy_(obs)
        self.actions[idxs].copy_(actions)
        self.masks[idxs].copy_(masks)
        self.advantages[idxs].copy_(advantages)
        self.ptr = (self.ptr + n) % self.max_size

        self.size = min(self.size + n, self.max_size)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(obs=self.obs[idxs],
                     actions=self.actions[idxs],
                     advantages=self.advantages[idxs],
                     masks=self.masks[idxs])

        # permute tensor to shape n_steps x batch_size x shape
        return {k: v.permute(1, 0, 2) for k, v in batch.items()}

    def sample_most_recent(self):
        if self.size < self.batch_size:
            idxs = np.arange(0, self.size)
        else:
            idxs = np.arange(self.ptr - self.batch_size,
                             self.ptr) % self.max_size
        # the first n_procecsses indexes will be used to storage current rollout
        # the rest are most recent rollouts
        idxs = np.concatenate(
            [
                idxs[-self.n_processes:],
                idxs[:-self.n_processes]
            ]
        )
        batch = dict(obs=self.obs[idxs],
                     actions=self.actions[idxs],
                     advantages=self.advantages[idxs],
                     masks=self.masks[idxs])

        # permute tensor to shape n_steps x batch_size x shape
        return {k: v.permute(1, 0, 2) for k, v in batch.items()}
