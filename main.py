import os
import time
from collections import deque

import numpy as np
import torch

from core import algorithms, utils
from core.arguments import get_args
from core.envs import make_vec_envs
from core.agents import Policy, MetaInputDependentPolicy
from core.storage import RolloutStorage
from core.agents.models import BNCNN
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    if not args.use_proper_time_limits:
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, args.num_frame_stack, args=args)
    else:
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, True, args.num_frame_stack, args.max_episode_steps, args=args)
    
    if args.algo == 'idp_a2c':
        actor_critic = MetaInputDependentPolicy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy},
            num_inner_steps=args.num_inner_steps,
            adapt_lr=args.adapt_lr)
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algorithms.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppo':
        agent = algorithms.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algorithms.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
    elif args.algo == 'idp_a2c':
        agent = algorithms.MetaInputDependentA2C(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm
        )
    else:
        raise ValueError("Not Implemented algorithm...")

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    # lehduong: Total number of gradient updates
    # basically, the agent will act on the environment for a number of steps
    # which is usually referred to as n_steps. Then, we compute the cummulative
    # reward, update the policy. After that, we continue rolling out the agent 
    # in the environment repeatedly. If the trajectory is ended, simply reset it and    # do it again.
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    # the gradient update interval to increase number of stream jobs
    curriculum_interval = int(num_updates/args.num_curriculum_time)

    for j in range(num_updates):
        if (args.env_name == 'load_balance') and (j % curriculum_interval) == 0:
            args.num_stream_jobs = int(args.num_stream_jobs * args.num_stream_jobs_factor)
            # reconstruct environments to increase the number of stream jobs 
            # also alter the random seed
            if not args.use_proper_time_limits:
                envs = make_vec_envs(args.env_name, args.seed+j, args.num_processes,
                         args.gamma, args.log_dir, device, False, args.num_frame_stack, args=args)
            else:
                envs = make_vec_envs(args.env_name, args.seed+j, args.num_processes,
                         args.gamma, args.log_dir, device, True, args.num_frame_stack, args.max_episode_steps, args=args)
            print("Increase the number of stream jobs to "+str(args.num_stream_jobs))
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)

        # decrease learning rate linearly
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        
        # Rolling out, collecting and storing SARS (State, action, reward, new state)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            # TODO: park env does not support cuda tensor???
            obs, reward, done, infos = envs.step(action.cpu())
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "\nUpdates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            print(
                " Value loss: {:.2f} Action loss {:2f} Dist Entropy {:2f}"
                .format(value_loss,
                        action_loss,
                        dist_entropy)
            )
        
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            # alter the random seed
            evaluate(actor_critic, args.env_name, args.seed+j,
                     args.num_processes, eval_log_dir, device, env_args=args)


if __name__ == "__main__":
    main()
