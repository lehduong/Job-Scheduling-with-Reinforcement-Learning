import os
import time
import numpy as np
import torch
import os.path as osp

from torch import optim
from collections import deque
from itertools import chain
from tensorboardX import SummaryWriter

from core import algorithms, utils
from core.agents import Policy, MetaInputDependentPolicy
from core.agents.heuristic.load_balance import ShortestProcessingTimeAgent, LeastWorkAgent
from core.arguments import get_args
from core.envs import make_vec_envs
from core.storage import RolloutStorage
from evaluation import evaluate


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
    if args.algo.startswith('idp'):
        # actor critic for input-dependent baseline
        # i.e. meta critic (and conventional actor)
        actor_critic = MetaInputDependentPolicy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy},
            num_inner_steps=args.num_inner_steps,
            adapt_lr=args.adapt_lr)
    else:
        # vanilla actor-critic
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})

    if args.resume_dir is not None:
        print("=> Resuming from checkpoint: {}".format(args.resume_dir))
        actor_critic = torch.load(args.resume_dir, map_location='cpu')[0]
    actor_critic.to(device)

    optimizer = optim.Adam(
        actor_critic.parameters(),
        args.actor_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.5,
                                                     min_lr=1e-5,
                                                     patience=500,
                                                     threshold=0.1,
                                                     verbose=True)

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

    expert = ShortestProcessingTimeAgent(args.load_balance_service_rates)

    for j in range(num_updates):
        random_seed = args.seed if args.fix_job_sequence else args.seed + j

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

        # update
        obs_shape = rollouts.obs.size()[2:]

        # imitation learning
        imitation_loss, accuracy = actor_critic.imitation_learning(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            expert)

        optimizer.zero_grad()

        imitation_loss.backward()

        optimizer.step()
        scheduler.step(imitation_loss)

        results = {
            'imitation loss': imitation_loss,
            'accuracy': accuracy
        }

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
            print("Updates {}, num timesteps {}, FPS {}"
                  "\n=> Last {} training episodes: mean/median reward "
                  "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(
                      j, total_num_steps,
                      int(total_num_steps / (end - start)),
                      len(episode_rewards), np.mean(episode_rewards),
                      np.median(episode_rewards), np.min(episode_rewards),
                      np.max(episode_rewards)))
            result_str = "=> "
            for k, v in results.items():
                result_str = result_str + "{}: {:.2f} ".format(k, v)
            print(result_str)

            writer.add_scalar("train/reward", np.mean(episode_rewards), j)

        writer.add_scalar('train/loss', imitation_loss, j)
        writer.add_scalar('train/accuracy', accuracy, j)

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
