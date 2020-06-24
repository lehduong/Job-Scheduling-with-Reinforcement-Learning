"""
Make environments of Park Platform
"""
import torch
import numpy as np
import os
import park
import gym
import random

from park.spaces.box import Box
from baselines import bench, logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.wrappers import TimeLimit
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

from .load_balance_wrappers import ProcessLoadBalanceObservation, \
    LoadBalanceRandomReset, RewardNormalize, FixJobSequence


PARK_ENV_LIST = ['spark', 'spark_sim',
                 'load_balance']


def make_env(env_id,
             seed,
             rank,
             log_dir,
             allow_early_resets,
             max_episode_steps=None,
             args=None,
             train=True):
    def _thunk():
        if env_id not in PARK_ENV_LIST:
            raise ValueError("Unsupported environment, expect the environment to be one of "
                             + str(PARK_ENV_LIST)+" but got: "+str(env_id))
        elif env_id == 'load_balance':
            # arrange the number of stream jobs
            env = park.make(env_id,
                            num_stream_jobs=args.num_stream_jobs,
                            service_rates=args.load_balance_service_rates)

            # random act after resetting to diversify the state
            # only use when training
            if train:
                env = LoadBalanceRandomReset(
                    env, args.max_random_init_steps)

            # if using load balance, clip and normalize the observation with this wrapper
            if args is not None:
                env = ProcessLoadBalanceObservation(env,
                                                    args.job_size_norm_factor,
                                                    args.server_load_norm_factor,
                                                    args.highest_server_obs,
                                                    args.highest_job_obs,
                                                    args.elapsed_time_norm_factor,
                                                    args.highest_elapsed_time
                                                    )
                # normalize reward
                env = RewardNormalize(env, args.reward_norm_factor)

                if args.fix_job_sequence:
                    # fix job sequence
                    env = FixJobSequence(env, seed)

        if max_episode_steps:
            env = TimeLimit(env, max_episode_steps)
            # adding information to env for computing return
            env = TimeLimitMask(env)

        # IMPORTANT: all environments used same random seed to repeat the input-process
        if train:
            env.seed(seed)
        else:
            env.seed(seed + rank)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  log_dir,
                  device,
                  allow_early_resets,
                  max_episode_steps=None,
                  args=None,
                  train=True):
    """
        Make vectorized environments 
        :param env_name: str - name of environment
        :param seed: int - random seed of environment
        :num_process: int - number of parallel environment
        :param log_dir: str - path to log directory
        :param device: str - 'cuda' or 'cpu'
        :param allow_early_reset: bool - if apply TimeLimitMask on environments, set this param to True
        :param max_episode_steps: int - maximum number of action in 1 episode
        :param args: ArgsParser - use to specifiy environment args
        :param train: bool - determine if we are using created to train or evaluate
                            if we're training, all environment share same random seed to repeat input sequence
                            otherwise, we diversify the random seed
    """
    envs = [
        make_env(env_id=env_name, seed=seed, rank=i, log_dir=log_dir,
                 allow_early_resets=allow_early_resets,
                 max_episode_steps=max_episode_steps, args=args, train=train)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    return envs


def load_balance_states_to_inputs(states):
    """
        Transform states of LoadBalance Env to inputs sequences
        :param states: torch.Tensor of shape T x N_processes x (Num_servers + 2)
        :return: torch.Tensor of shape T x N_processes x 2
    """
    return states[:, :, -2:]


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
