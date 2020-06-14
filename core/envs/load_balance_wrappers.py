import gym
import random
import numpy as np


class ProcessLoadBalanceObservation(gym.ObservationWrapper):
    """
        Normalize and clip the observation of LoadBalance environment
        :param job_size_norm_factor: float - divide job_size by this factor
        :param highest_server_obs: float - clip the server (in observation) having load higher than this value
        :param highest_job_obs: float - clip the job (in observation) having size greater than this value
    """

    def __init__(self, env, job_size_norm_factor, server_load_norm_factor, highest_server_obs, highest_job_obs):
        super().__init__(env)
        self.job_size_norm_factor = job_size_norm_factor
        self.server_load_norm_factor = server_load_norm_factor
        self.highest_server_obs = highest_server_obs
        self.highest_job_obs = highest_job_obs

        # compute clip threshold
        num_server = len(env.servers)
        self.threshold = np.array(
            [self.highest_server_obs]*num_server+[self.highest_job_obs])
        # compute the normalize vector
        self.norm_vec = np.array(
            [self.server_load_norm_factor]*num_server+[self.job_size_norm_factor])

    def observation(self, observation):
        # normalized
        observation = observation/self.norm_vec
        return np.minimum(observation, self.threshold)


class LoadBalanceRandomReset(gym.Wrapper):
    def __init__(self, env, num_random_steps_reset=50):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.num_random_steps_reset = num_random_steps_reset

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        for _ in range(self.num_random_steps_reset):
            obs, _, done, _ = self.env.step(
                random.randint(0, len(self.env.servers)-1))
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class RewardNormalize(gym.RewardWrapper):
    """
        Divide the reward by a fixed value
    """

    def __init__(self, env, norm_factor):
        super().__init__(env)
        self.norm_factor = norm_factor

    def reward(self, reward):
        return reward/self.norm_factor
