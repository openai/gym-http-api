# Importing required modules
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import functools
from operator import mul

# Importing necessary PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import PPO code
import helpers.ppo as ppo
import helpers.utils as utils
from helpers.arguments import get_args
from helpers.envs import make_vec_envs
from helpers.model import Policy
from helpers.storage import RolloutStorage
from evaluation import evaluate

# For log file
from datetime import datetime

args = get_args()

if __name__ == '__main__':
    # Main program starts here

    # For some reason, this seems necessary for a wrapper that allows early environment termination
    log_dir = os.path.expanduser('/tmp/gym/') # Nothing is really stored here
    utils.cleanup_log_dir(log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args.env_name, args.env_state, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, allow_early_resets=True)

    # Generalize this
    model_path = "models/2019-07-13-GreenHillZone.Act1-lamarck/gen4/model-0.pt"
    if torch.cuda.is_available() :
        actor_critic, ob_rms = torch.load(model_path)
    else :
        actor_critic, ob_rms = torch.load(model_path, map_location='cpu')

    # Agent is initialized
    agent = ppo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=1e-8, # This epsilon is not for exploration. It is for numerical stability of the Adam optimizer. This is the default value.
        max_grad_norm=args.max_grad_norm)
            
    print("Evaluating.", end=" ")
    fitness, behavior_char = evaluate(agent.actor_critic, envs, device, 0, args)
    print("({},{})".format(fitness, behavior_char))