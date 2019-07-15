# Importing required modules
import os

# Importing necessary PyTorch stuff
import torch

# Import PPO code
import helpers.utils as utils
from helpers.arguments import get_args
from helpers.envs import make_vec_envs
from evaluation import evaluate

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
            
    print("Evaluating.", end=" ")
    fitness, behavior_char = evaluate(actor_critic, envs, device, 0, args)
    print("({},{})".format(fitness, behavior_char))