import numpy as np
import torch

#from helpers import utils
#from helpers.envs import make_vec_envs

# Will start using this method for evaluation. Every trail will involve some learning,
# but the actual fitness and behavior characterizations will be based on this method,
# where the learning is turned off. This prevents random epsilon exploration moves
# from unfairly huring the agent's fitness

# Evaluates the agent without any learning occurring
# This is the original header:
#def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir, device):
# Modified
def evaluate(actor_critic, eval_envs, device, generation, args):
    # Removed this. The only difference seems to be that they didn't use a discount factor/gamma, but they probably should.
    #eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
    #                          None, eval_log_dir, device, True)

    # Removed this too. Not sure what this does, but ob_rms was causing problems.
    #vec_norm = utils.get_vec_normalize(eval_envs)
    #if vec_norm is not None:
    #    vec_norm.eval()
    #    vec_norm.ob_rms = ob_rms

    watching = args.watch_frequency is not None and generation % args.watch_frequency == 0
    if watching:
        print("Watching.", end = " ")

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)

    # Schrum: Added for tracking behavior characterization
    behaviors = []

    # Evaluates 10 times and gets the average
    # TODO: Problem! The behavior characterization has 10 distinct episodes in it now.
    #       This means characterizations from separate episodes won't line up, making the
    #       comparison meaningless! Need to fix! For now, just make the number of episodes = 1
    #while len(eval_episode_rewards) < 10:
    while len(eval_episode_rewards) < 1: # TODO: Change this eventually
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # For observing
        if watching: 
            eval_envs.render()
        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        # Schrum: Added to make behavior charaterization
        info = infos[0] 
        if(len(infos) > 1): # Is the length really always 1?
            print("infos too long")
            print(infos)
            quit()
        xpos = info['x']
        ypos = info['y']
        behaviors.append(xpos)
        behaviors.append(ypos)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    mean_return = np.mean(eval_episode_rewards)
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), mean_return))

    return mean_return, behaviors
