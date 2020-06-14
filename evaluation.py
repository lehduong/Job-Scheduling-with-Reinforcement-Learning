import numpy as np
import torch
import copy

from core import utils
from core.envs import make_vec_envs
from core.agents.heuristic.load_balance import LeastWorkAgent, \
    ShortestProcessingTimeAgent, RandomAllocateAgent, EarliestCompletionTimeAgent


def evaluate(actor_critic, env_name, seed, num_processes, eval_log_dir,
             device, env_args=None):
    returns = benchmark_heuristic([LeastWorkAgent(),
                                   ShortestProcessingTimeAgent(),
                                   RandomAllocateAgent(),
                                   EarliestCompletionTimeAgent()],
                                  env_name=env_name,
                                  seed=seed,
                                  num_processes=num_processes,
                                  log_dir=eval_log_dir,
                                  device=device,
                                  args=env_args)
    # benchmark heuristic
    # least_work
    eval_envs = make_vec_envs(env_name=env_name,
                              seed=seed + num_processes,
                              num_processes=num_processes,
                              log_dir=eval_log_dir,
                              device=device,
                              allow_early_resets=True,
                              train=False,
                              args=env_args)
    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    # TODO: Deterministic configuration results in much worse performance \
    # compare to non-deterministic one
    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)

        # Obser reward and next obs
        # FIXME: debug why actions must be moved to cpu?
        obs, _, done, infos = eval_envs.step(action.cpu())

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()
    returns['RLAgent'] = eval_episode_rewards

    # print out the result
    for k, v in returns.items():
        print("=> Evaluate {} using {} episodes: mean reward {:.5f}\n".format(
            k, len(v), np.mean(v)))
    return returns


def benchmark_single_heuristic(agent, eval_envs):
    """
        Compute return of a single heuristic agent
    """
    obs = eval_envs.reset()
    eval_episode_rewards = []

    while len(eval_episode_rewards) < 10:
        action = agent.act(obs)
        # Obser reward and next obs

        obs, _, done, infos = eval_envs.step(action.cpu())

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return eval_episode_rewards


def benchmark_heuristic(agents, **kwargs):
    """
        Compute return of all heuristics
    """
    ret = {}
    for agent in agents:
        envs = make_vec_envs(env_name=kwargs['env_name'],
                             seed=kwargs['seed'] + kwargs['num_processes'],
                             num_processes=kwargs['num_processes'],
                             log_dir=kwargs['log_dir'],
                             device=kwargs['device'],
                             allow_early_resets=True,
                             train=False,
                             args=kwargs['args'])

        eval_episode_rewards = benchmark_single_heuristic(agent, envs)
        # append the result to return dictionary
        ret[agent.__class__.__name__] = eval_episode_rewards

    return ret
