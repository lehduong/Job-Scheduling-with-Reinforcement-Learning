import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='idp_a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--critic-lr', type=float, default=5e-4, help='learning rate of critic (default: 1e-3)')
    parser.add_argument(
        '--actor-lr', type=float, default=5e-4, help='learning rate of actor (default: 1e-3)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=4,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1000,
        help='number of forward steps in A2C (default: 20)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='load_balance',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='ex',
        help='directory to save agent logs (default: ex)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--max-episode-steps',
        default=1000,
        type=int,
        help='maximum number of steps per episode of environment (default: 1000)')
    parser.add_argument(
        '--num-frame-stack',
        default=1,
        help='number of observation that will be grouped together (default: 4)')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    
    # META INPUT-DEPENDENT BASELINE
    parser.add_argument(
        '--num-inner-steps',
        type=int,
        default=10,
        help='number of gradient steps for adapting to new input sequences (default: 4)')
    parser.add_argument(
        '--adapt-lr',
        type=float,
        default=1e-2,
        help='learning rate of innerloop when adapting to new input sequences (default: 2e-3)')

    # LOAD BALANCE ENVIRONMENT
    parser.add_argument(
        '--num-curriculum-time',
        default=65,
        type=int,
        help='number of time we would like to increase the num-stream-jobs in load balance env (default: 65)')
    parser.add_argument(
        '--num-stream-jobs-factor',
        default=1.1,
        type=float,
        help='exponentially increase the number of stream jobs in environment after some interval (default: 1.1)')
    parser.add_argument(
        '--job-size-norm-factor',
        default=10,
        type=float,
        help='normalize factor of job size in load balance env (default: 1000)')
    parser.add_argument(
        '--server-load-norm-factor',
        default=50,
        type=float,
        help='normalize factor of server load in load balance env (default: 5000)')
    parser.add_argument(
        '--highest-server-obs',
        default=2000,
        type=float,
        help='Clip server having higher load than this value in load balance environment (default: 20)')
    parser.add_argument(
        '--highest-job-obs',
        default=1000,
        type=float,
        help='Clip job having greater size than this value in load balance environment (default: 10)')
    parser.add_argument(
        '--reward-norm-factor',
        default=1000,
        type=float,
        help='normalize factor of reward in training (default: 1000)')
    parser.add_argument(
        '--num-stream-jobs',
        default=1000,
        type=float,
        help='number of stream jobs of load balance env in training (default: 1000)')
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr', 'idp_a2c']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo', 'idp_a2c'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
