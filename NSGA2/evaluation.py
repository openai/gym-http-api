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
        print("Watching.", end=" ")

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)

    # Schrum: Added for tracking behavior characterization
    behaviors = []
    steps_without_change_in_x = 0
    last_x = 0
    accumulated_reward = 0
    step = 0

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
        if device.type == 'cuda':  # For some reason, CUDA actions are nested in an extra layer
            action = action[0]
        obs, reward, done, infos = eval_envs.step(action)
        # Would prefer to get reward from info['episode']['r'], but this is only filled if episode ends normally.
        # Need to track it manually if eval ends prematurely
        accumulated_reward += reward[0][0].item() # * (args.gamma ** step)
        step += 1

        # Schrum: Added to make behavior characterization
        info = infos[0] 
        xpos = info['x']
        ypos = info['y']
        if args.final_pt:
            if done:
                behaviors.extend([xpos, ypos])
        else:
            behaviors.extend([xpos, ypos])

        if xpos == last_x:
            steps_without_change_in_x += 1
            if steps_without_change_in_x >= args.eval_progress_fail_time:
                if args.final_pt:
                    behaviors.extend([xpos, ypos])
                print("End Early, stuck at {} for {} steps.".format(xpos, steps_without_change_in_x), end=" ")
                # Artificially set accumulated reward to end evaluation early
                info['episode'] = {'r': accumulated_reward}
        else:
            steps_without_change_in_x = 0

        last_x = xpos  # remember previous x position

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                # Confirmed that the calculations are the same, except for rounding errors (double vs float?)
                #if accumulated_reward != info['episode']['r']:
                #    print("CALCULATED REWARDS AND REPORTED REWARDS DO NOT MATCH! {} != {}".format(accumulated_reward, info['episode']['r']))
                accumulated_reward = 0
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    mean_return = np.mean(eval_episode_rewards)
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), mean_return))

    return mean_return, behaviors
