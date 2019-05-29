from argparse import ArgumentParser

# default arguments
EPSILON = 0.2
EPSILON_STEP = 10
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.001
GAMMA = 0.999
NEPISODE = 1000

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('algorithm', metavar='ALG', help='The algorithm you wish to ' + \
                'work with (currently only A2C, DQN, and PG).')
    parser.add_argument('domain', metavar='DOMAIN', help='The OpenAI Gym domain you ' + \
                'wish to work in (currently only discrete).')
    parser.add_argument('-e', '--eps', type=float,
            dest='epsilon', help='DQN: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--ep_s', type=float,
            dest='epsilon_step', help='DQN: epsilon step (default %(default)s)',
            metavar='EPSILON_STEP', default=EPSILON_STEP)
    parser.add_argument('--ep_d', type=float,
            dest='epsilon_decay', help='DQN: epsilon decay (default %(default)s)',
            metavar='EPSILON_DECAY', default=EPSILON_DECAY)
    parser.add_argument('--ep_m', type=float,
            dest='epsilon_min', help='DQN: minimum epsilon (default %(default)s)',
            metavar='EPSILON_MIN', default=EPSILON_MIN)
    parser.add_argument('-g', '--gamma', type=float,
            dest='gamma', help='Gamma parameter (default %(default)s)',
            metavar='GAMMA', default=GAMMA)
    parser.add_argument('--nepisode', type=int,
            dest='nepisode', help='Number of episodes (default %(default)s)',
            metavar='NEPISODE', default=NEPISODE)
    return parser

#def lrq = 0.02, gamma = 0.999, epsilon = 0.2, estep = 10, edecay = 0.99, emin = 0.001