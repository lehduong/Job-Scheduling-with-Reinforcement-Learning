import os
import time
import numpy as np
import torch
import os.path as osp

from collections import deque
from core import algorithms, utils
from core.agents import Policy
from core.agents.heuristic.load_balance import ShortestProcessingTimeAgent, \
    EarliestCompletionTimeAgent
from core.arguments import get_args
from core.envs import make_vec_envs
from core.storage import RolloutStorage
from evaluation import evaluate
from tensorboardX import SummaryWriter


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    base_dir = osp.expanduser(args.log_dir)
    log_dir = osp.join(base_dir, 'train_log')
    eval_log_dir = osp.join(base_dir, "eval_log")
    tensorboard_dir = osp.join(base_dir, "tensorboard_log")

    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    utils.cleanup_log_dir(tensorboard_dir)
    utils.dump_config(args, osp.join(base_dir, 'config.txt'))

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    writer = SummaryWriter(tensorboard_dir)

    # limited the number of steps for each episode
    # IMPORTANT: for load balance / spark-sim we automatically do this by setting
    # the number of stream jobs
    if not args.use_proper_time_limits:
        envs = make_vec_envs(env_name=args.env_name,
                             seed=args.seed,
                             num_processes=args.num_processes,
                             log_dir=log_dir,
                             device=device,
                             allow_early_resets=False,
                             args=args)
    else:
        envs = make_vec_envs(env_name=args.env_name,
                             seed=args.seed,
                             num_processes=args.num_processes,
                             log_dir=log_dir,
                             device=device,
                             allow_early_resets=True,
                             max_episode_steps=args.max_episode_steps,
                             args=args)

    # create actor critic
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    # if the resume directory is provided, then directly load that checkpoint
    if args.resume_dir is not None:
        print("=> Resuming from checkpoint: {}".format(args.resume_dir))
        actor_critic = torch.load(args.resume_dir, map_location='cpu')[0]
    actor_critic.to(device)

    # expert for imitation learning
    if args.use_imitation_learning:
        expert = EarliestCompletionTimeAgent(args.load_balance_service_rates)
    else:
        expert = None

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
        agent = algorithms.MIB_A2C(
            actor_critic,
            args.entropy_coef,
            lr=args.lr,
            adapt_lr=args.adapt_lr,
            num_inner_steps=args.num_inner_steps,
            max_grad_norm=args.max_grad_norm,
            expert=expert,
            il=args.il_coef
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

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    # the gradient update interval to increase number of stream jobs
    curriculum_interval = int(num_updates / args.num_curriculum_time)

    for j in range(num_updates):
        random_seed = args.seed if args.fix_job_sequence else args.seed + j
        # if using load_balance environment: \
        # we have to gradually increase number of stream jos
        if (args.env_name == 'load_balance') and ((j + 1) % curriculum_interval) == 0:
            args.num_stream_jobs = int(
                args.num_stream_jobs * args.num_stream_jobs_factor)

            # reconstruct environments to increase the number of stream jobs
            # also alter the random seed
            if not args.use_proper_time_limits:
                envs = make_vec_envs(env_name=args.env_name,
                                     seed=random_seed,
                                     num_processes=args.num_processes,
                                     log_dir=log_dir,
                                     device=device,
                                     allow_early_resets=False,
                                     args=args)
            else:
                envs = make_vec_envs(env_name=args.env_name,
                                     seed=random_seed,
                                     num_processes=args.num_processes,
                                     log_dir=log_dir,
                                     device=device,
                                     allow_early_resets=True,
                                     max_episode_steps=args.max_episode_steps,
                                     args=args)

            print("Increase the number of stream jobs to " +
                  str(args.num_stream_jobs))
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)

        # decrease learning rate linearly
        if args.use_linear_lr_decay:
            cur_lr = utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        else:
            cur_lr = agent.optimizer.param_groups[0]["lr"]

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
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        results = agent.update(rollouts)

        rollouts.after_update()

        # SAVE trained model
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

        # LOG TRAINING results
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("="*90)
            print("Updates {}, num timesteps {}, FPS {}, LR: {}"
                  "\n=> Last {} training episodes: mean/median reward "
                  "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(
                      j, total_num_steps,
                      int(total_num_steps / (end - start)),
                      cur_lr,
                      len(episode_rewards), np.mean(episode_rewards),
                      np.median(episode_rewards), np.min(episode_rewards),
                      np.max(episode_rewards)))
            result_str = "=> "
            for k, v in results.items():
                result_str = result_str + "{}: {:.2f} ".format(k, v)
            print(result_str)

            writer.add_scalar("train/reward", np.mean(episode_rewards), j)

        # EVALUATE performance of learned policy along with heuristic
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            # alter the random seed
            eval_results = evaluate(actor_critic, args.env_name, random_seed,
                                    args.num_processes, eval_log_dir, device, env_args=args)
            writer.add_scalars(
                'eval/reward',
                {k: np.mean(v) for k, v in eval_results.items()},
                j)

    writer.close()


if __name__ == "__main__":
    main()
