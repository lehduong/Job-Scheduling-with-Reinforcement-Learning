import numpy as np
import torch
import copy

from core import utils
from core.envs import make_vec_envs
from core.agents.heuristic.load_balance import LeastWorkAgent, ShortestProcessingTimeAgent

def evaluate(actor_critic, env_name, seed, num_processes, eval_log_dir,
             device, env_args=None):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, train=False, args=env_args)

    # benchmark heuristic
    # least_work
    leastwork_env = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, train=False, args=env_args)
    benchmark_heuristic(LeastWorkAgent(), leastwork_env)
    # shortest processing time
    spt_env = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, train=False, args=env_args)
    benchmark_heuristic(ShortestProcessingTimeAgent(), spt_env)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        #TODO: Park doesn't support GPU tensor
        obs, _, done, infos = eval_envs.step(action.cpu())

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

def benchmark_heuristic(agent, eval_envs):
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

    print(" Evaluation" + agent.__class__.__name__ + " using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
