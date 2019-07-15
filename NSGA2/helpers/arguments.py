import argparse
import torch

'''
usage: NSGAII.py [-h] [--evol-mode EVOL_MODE] [--num_gens NUM_GENOMES]
                 [--pop_size POP_SIZE] [--lr LR] [--eps EPS] [--gamma GAMMA]
                 [--use-gae] [--gae-lambda GAE_LAMBDA]
                 [--entropy-coef ENTROPY_COEF]
                 [--value-loss-coef VALUE_LOSS_COEF]
                 [--max-grad-norm MAX_GRAD_NORM] [--seed SEED]
                 [--cuda-deterministic] [--num-processes NUM_PROCESSES]
                 [--num-steps NUM_STEPS] [--ppo-epoch PPO_EPOCH]
                 [--num-mini-batch NUM_MINI_BATCH] [--clip-param CLIP_PARAM]
                 [--log-interval LOG_INTERVAL] [--save-interval SAVE_INTERVAL]
                 [--eval-interval EVAL_INTERVAL]
                 [--num-env-steps NUM_ENV_STEPS] [--env-name ENV_NAME]
                 [--env-state ENV_STATE] [--log-dir LOG_DIR]
                 [--save-dir SAVE_DIR] [--no-cuda] [--use-proper-time-limits]
                 [--recurrent-policy] [--use-linear-lr-decay]
'''

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--evol-mode',
        default='baldwin',
        help='evolution mode: none | baldwin | lamarck')
    parser.add_argument(
        '--watch-frequency',
        type=int, default=9999, # Set larger to max int value?
        help='set to indicate number of generations between rendering of evaluation')
    parser.add_argument('--eval-progress-fail-time',
                        type=int, default=50, help='Number of steps without a change in x coordinate before Sonic evaluation terminates (default: 50)')
    parser.add_argument('--num-gens',
                        type=int, default=50, help='number of genomes to run through (default: 50)')
    parser.add_argument('--num-updates',
                        type=int, default=128, help='number of learning updates if learning is used (default: 128)')
    parser.add_argument('--pop-size',
                        type=int, default=10, help='population size per genome (default: 10)')
    parser.add_argument('--lr',
                        type=float, default=2e-4, help='learning rate (default: 2e-4)')
    parser.add_argument('--gamma',
                        type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--watch-learning',
        action='store_true',
        default=False,
        help='Render visualization during learning')
    parser.add_argument('--gae-lambda',
                        type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef',
                        type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef',
                        type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm',
                        type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed',
                        type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--init-from-network',
        action='store_true',
        default=True,
        help="Genome initialization based on network weight initializer")
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes',
                        type=int, default=1, help='how many training CPU processes to use (default: 1)')
    parser.add_argument('--num-steps',
                        type=int, default=1024, help='number of forward steps (default: 1024)')
    parser.add_argument('--ppo-epoch',
                        type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch',
                        type=int, default=16, help='number of batches for ppo (default: 16)')
    parser.add_argument('--clip-param',
                        type=float, default=0.1, help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--log-interval',
                        type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval',
                        type=int, default=1, help='save interval, one save per n generations (default: 1)')
    parser.add_argument('--eval-interval',
                        type=int, default=None, help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--num-env-steps',
                        type=int, default=10e6, help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='SonicTheHedgehog-Genesis',
        help='Sonic environment to train on (default: SonicTheHedgehog-Genesis)')
    parser.add_argument(
        '--env-state',
        default='GreenHillZone.Act1',
        help='state of given environment (default: GreenHillZone.Act1)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--final-pt',
        action='store_true',
        default=True,
        help='log only an agent\'s final point to a behavior characterization')
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
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=True,
        help='use a linear schedule on the learning rate')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.evol_mode in ['none', 'baldwin', 'lamarck']

    return args
