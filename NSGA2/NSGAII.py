# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

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

args = get_args()

# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front


# Given all behavior characterizations, compute all novelty scores
def calculate_novelty(behavior_characterizations):
    ns = []
    for i in range(0, len(behavior_characterizations)):
        total_dist = 0
        for j in range(0, len(behavior_characterizations)):
            # Compare to all others except self
            if i != j:
                i_behavior = behavior_characterizations[i]
                j_behavior = behavior_characterizations[j]
                dist = euclidean_dist(i_behavior, j_behavior)
                total_dist += dist
        avg_dist = total_dist / len(behavior_characterizations)
        ns.append(avg_dist)
    return ns


# Distance between two behavior characterizations
def euclidean_dist(i_behavior, j_behavior):
    index = total = 0
    # Position lists could be different lengths
    minimum = min(len(i_behavior), len(j_behavior))
    # Difference between common positions
    while index < minimum: 
        total += (i_behavior[index] - j_behavior[index]) ** 2
        index += 1
    # If i_behavior is longer
    while index < len(i_behavior):
        total += i_behavior[index] ** 2
        index += 1
    # If j_behavior is longer
    while index < len(j_behavior):
        total += j_behavior[index] ** 2
        index += 1
    # NOTE: Could there be an issue where both vectors being compared are shorter than the longest
    #       vector in the whole population? What would be the consequences?
    total = math.sqrt(total)
    return total


# Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance


# Function to carry out the crossover
def crossover(a, b):
    r = random.random()
    if r > 0.5: # Crossover by adding the vectors and averaging
        return mutation((a+b)/2)
    else: # No crossover ... just mutate first of the two
        return mutation(a)


# Function to carry out the mutation operator
def mutation(solution):
    #print("Original: ", solution)
    # TODO: Make the mutation range be an args parameter?
    max_range = 1
    # Bit vector multiplied by the range
    mutationScale = np.random.randint(2, size=len(solution)) * max_range
    solution = np.random.normal(solution, mutationScale)
    #print("Mutated : ", solution)
    return solution

# One network learns from evolved starting point.
def learn(env, agent):
    global device

    net = agent.actor_critic
    num_steps = args.num_steps
    rollouts = RolloutStorage(num_steps, args.num_processes, envs.observation_space.shape,
                              envs.action_space, net.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    done = False
    num_updates = 5 # Keep value small since we evaluate for multiple episodes
    for j in range(num_updates):
    # while True:  # Until the episode is over

        # if use_linear_lr_decay:
        # decrease learning rate linearly
        # simply changes learning rate of optimizer
        
        # Disable this for now ... may want to replace j with the current generation or some other value
        # utils.update_linear_schedule(
        #    agent.optimizer, j, num_updates,
        #    learning_rate)

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = net.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # To watch while learning
            # envs.render()
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        #print("Finished {} steps".format(num_steps))

        with torch.no_grad():
            next_value = net.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        # These variables were only used for logging in the original PPO code.
        # However, the agent.update command is important, and is doing the learning.
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

def random_genome(n):
    # n is the number of weights
    return np.random.uniform(-1, 1, n).astype(np.float32)

# Evaluate every member of the included population, which is a collection
# of weight vectors for the neural networks.
def evaluate_population(solutions, agent, generation):
    global device
    fitness_scores = []
    behavior_characterizations = []

    for i in range(pop_size):
        print("Evaluating genome #{}:".format(i), end=" ")  # No newline: Fitness will print here

        # Creates solutions[i] Tensor and converts it to type Float before passing on to set_weights
        weights = torch.from_numpy(solutions[i])
        weights = weights.to(device)
        set_weights(agent.actor_critic, weights)

        if args.evol_mode in {'baldwin', 'lamarck'}:        
            # Make the agent optimize the starting weights. Weights of agent are changed via side-effects
            print("Learning.", end=" ")
            learn(envs, agent)

            # TODO: If mode is lamarck, then copy the model weights back into solutions[i]

        # Do evaluation of agent without learning to get fitness and behavior characterization
        ob_rms = None # utils.get_vec_normalize(envs).ob_rms # Not sure what this is. From gym-http-api\pytorch-a2c-ppo-acktr-gail\main.py
        seed = args.seed  # TODO: Probably what the random seed to be different each time
        num_processes = args.num_processes  # TODO: Make command line param?
        # May want to change/remove the log dir of '/tmp/gym/'
        print("Evaluating.", end=" ")
        fitness, behavior_char = evaluate(agent.actor_critic, envs, device, generation, args)
        # print(fitness)
        # print(behavior_char)
        fitness_scores.append(fitness)
        behavior_characterizations.append(behavior_char)
            
    # Compare all of the behavior characterizations to get the diversity/novelty scores.
    novelty_scores = calculate_novelty(behavior_characterizations)
    # print(novelty_scores)
        
    return (fitness_scores, novelty_scores)

# Function to set the weights of the network to our randomly generated values.
def set_weights(net, weights):

    # Alex: this implementation initializes everything, including biases, to a random value.
    # I'll put some more work into this and see how I may go further (i.e. leaving biases as zeroes).
    i = 0

    # Get lengths of all the layers
    lengths = []
    sizes = []
    for layer in list(net.parameters()):
        length = torch.numel(layer)
        lengths.append(length)
        size = tuple(layer.size())
        sizes.append(size)

    # Split into several weight Tensors corresponding to the number of elements in every layer.
    split_vector = torch.split(weights, lengths, 0)

    # With every layer, replace its respective weight Tensor and set its data to that Tensor.
    for layer in list(net.parameters()):
        if i >= len(lengths) or i >= len(sizes):
            print("Index out of bounds")
            quit()
        if functools.reduce(mul, sizes[i], 1) != lengths[i]:
            print("Size error at Layer {}: {}".format(i, layer))
            quit()

        reshaped_weights = torch.reshape(split_vector[i], sizes[i])

        layer.data = reshaped_weights
        i += 1

if __name__ == '__main__':
    # Main program starts here

    # Competition version of environment make:
    # Mus pip install gym==0.12.1 in order for this to work
    # env = make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")

    log_dir = os.path.expanduser('/tmp/gym/')
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args.env_name, args.env_state, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, allow_early_resets=True)

    # Policy is created - in our case, since obs_shape is 3, it becomes a CNN
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'is_genesis': True})
    actor_critic.to(device)

    global num_weights

    # Agent is initialized
    agent = ppo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    # Schrum: Makes these small to test at first
    pop_size = args.pop_size

    # Initialization
    num_weights = sum(p.numel() for p in agent.actor_critic.parameters() if p.requires_grad)
    solutions = [random_genome(num_weights) for i in range(0, pop_size)]
    gen_no = 0
    while gen_no < args.num_gens:
        print("Start generation {}".format(gen_no))
        (fitness_scores, novelty_scores) = evaluate_population(solutions, agent, gen_no)

        # Display the fitness scores and novelty scores for debugging
        # for i in range(0,len(fitness_scores)):
        #     print("Fitness:",fitness_scores[i])
        #     print("Novelty:",novelty_scores[i])
        #     print("------------------")
        # print("+++++++++++++++++++++++++++++++++++++++++")
        
        print("Max, Average, Min Fitness are {} and {}".format(np.max(fitness_scores), np.mean(fitness_scores), np.min(fitness_scores)))
        print("Max, Average, Min Novelty are {} and {}".format(np.max(novelty_scores), np.mean(novelty_scores), np.max(novelty_scores)))
        non_dominated_sorted_solution = fast_non_dominated_sort(fitness_scores[:], novelty_scores[:])
        print("The best front for Generation number ", gen_no, " is")
        for valuez in non_dominated_sorted_solution[0]:
            print("Fitness:", fitness_scores[valuez])
            print("Novelty:", novelty_scores[valuez])
            print("------------------")
         
         
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(fitness_scores[:], novelty_scores[:], non_dominated_sorted_solution[i][:]))

        # The lambda children            
        solution2 = []
        # Generating offspring
        while len(solution2) != pop_size:
            a1 = random.randint(0, pop_size-1)
            b1 = random.randint(0, pop_size-1)
            solution2.append(crossover(solutions[a1], solutions[b1]))
            #print(solution2)

        print("Evaluate children of generation {}".format(gen_no))
        (fitness_scores2, novelty_scores2) = evaluate_population(solution2, agent, gen_no)
        # Combine parent and child populations into one before elitist selection
        function1_values2 = fitness_scores + fitness_scores2
        function2_values2 = novelty_scores + novelty_scores2

        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == pop_size:
                    break
            if len(new_solution) == pop_size:
                break
        # Combine the parent and child solutions so the best can be selected for the next parent population
        solution2 = solutions + solution2
        solutions = [solution2[i] for i in new_solution]
        gen_no += 1

    # Let's plot the final front now
    function1 = fitness_scores
    function2 = novelty_scores
    plt.xlabel('Fitness', fontsize=15)
    plt.ylabel('Novelty', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
