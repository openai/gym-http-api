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

# Importing necessary PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import PPO code
import helpers.ppo as ppo
import helpers.utils as utils
from helpers.envs import make_vec_envs
from helpers.model import Policy
from helpers.storage import RolloutStorage

# Use the Sonic contest environment
from retro_contest.local import make

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
    if r > 0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)


# Function to carry out the mutation operator
def mutation(solution):
    # Schrum: Just added these ranges. Appropriate?
    min_x = -3
    max_x = 3
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution


# Copied from Training.py in the sonicNEAT repo
def evaluate(env, net):
    global device
    fitness_current = 0
    behaviors = []

    learning_rate = 2.5e-4
    epsilon = 1e-5

    agent = ppo.PPO(
            net,
            clip_param=0.1,
            ppo_epoch=4,
            num_mini_batch=1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            lr=learning_rate,
            eps=epsilon,
            max_grad_norm=0.5)

    #for i in range(len(agent.actor_critic.base.main)):
    #    print(agent.actor_critic.base.main[i])
    #for p in agent.actor_critic.base.main.parameters():
    #    print(torch.numel(p))

    # for param in agent.actor_critic.base.main.parameters():
    #    print(param.data[0][0][0])
    #    param.data[0][0][0] = torch.FloatTensor([1,2,3,4,5,6,7,8])
    #    print(param.data[0][0][0])

    num_steps = 128
    num_processes = 1
    rollouts = RolloutStorage(num_steps, num_processes, envs.observation_space.shape, 
                              envs.action_space, net.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    
    done = False
    num_updates = 50
    for j in range(num_updates):
    #while True:  # Until the episode is over

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

            # Schrum: Uncomment this out to watch Sonic as he learns. This should only be done with 1 process though.
            #envs.render()
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # Schrum: This block is our code for fitness and behavior tracking
            fitness_current += reward
            info = infos[0]
            xpos = info['x']
            ypos = info['y']
            behaviors.append(xpos)
            behaviors.append(ypos)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            # Moved after the learning update above because we need to learn also (especially!) when Sonic dies  
            if done:
                print("DONE WITH EPISODE! Fitness = {}".format(fitness_current[0][0])) 
                #input("Press a key to continue") # Good for checking if fitness makes sense for the eval
                break

        with torch.no_grad():
            next_value = net.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae=True, gamma=0.99,
                                 gae_lambda=0.95, use_proper_time_limits=True)

        # These variables were only used for logging in the original PPO code
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
    
        if done: 
            print("DONE WITH EVAL!")
            break

    # For some reason, the individual fitness value is two layers deep in a tensor
    return fitness_current[0][0], behaviors


def random_genome(n):
    # n is the number of weights
    return np.random.rand(1, n)

# Evaluate every member of the included population, which is a collection
# of weight vectors for the neural networks.
def evaluate_population(solutions,net):
    fitness_scores = []
    behavior_characterizations = []

    for i in range(pop_size):
        print("Evaluating genome #{}".format(i))
        # Schrum: Need to set net weights based on a genome from population
        # Use solutions[i]
        fitness, behavior_char = evaluate(envs, net)
        # print(fitness)
        # print(behavior_char)
        fitness_scores.append(fitness)
        behavior_characterizations.append(behavior_char)
            
    # Compare all of the behavior characterizations to get the diversity/novelty scores.
    novelty_scores = calculate_novelty(behavior_characterizations)
    # print(novelty_scores)
        
    return (fitness_scores,novelty_scores)

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
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs("SonicTheHedgehog-Genesis", seed=1, num_processes=1,
                         gamma=0.99, log_dir='/tmp/gym/', device=device, allow_early_resets=True)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': True, 'is_genesis':True})
    actor_critic.to(device)

    # Schrum: Makes these small to test at first
    max_gen = 1
    pop_size = 10

    # Initialization
    # Schrum: This will need to be replaced with initialization for the network weights ... probably from -1 to 1, but how many will you need? Depends on network architecture.
    num_weights = 10  # What should this actually be?
    solution = [random_genome(num_weights) for i in range(0, pop_size)]
    gen_no = 0
    while gen_no < max_gen:        
        (fitness_scores,novelty_scores) = evaluate_population(solution,actor_critic)

        # Display the fitness scores and novelty scores for debugging
        # for i in range(0,len(fitness_scores)):
        #     print("Fitness:",fitness_scores[i])
        #     print("Novelty:",novelty_scores[i])
        #     print("------------------")
        # print("+++++++++++++++++++++++++++++++++++++++++")
        
        non_dominated_sorted_solution = fast_non_dominated_sort(fitness_scores[:], novelty_scores[:])
        print("The best front for Generation number ", gen_no, " is")
        for valuez in non_dominated_sorted_solution[0]:
            print("Fitness:", fitness_scores[valuez])
            print("Novelty:", novelty_scores[valuez])
            print("------------------")
         
         
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(fitness_scores[:], novelty_scores[:], non_dominated_sorted_solution[i][:]))
            
            
        # Schrum: The code below this point mutates a real-vector in the standard NSGA-II fashion, 
        #         which won't work with the NEAT networks we are currently playing with. However, when
        #         we switch to using CNNs, we will probably want this code (and have to change the code above).
            
        solution2 = solution[:]
        # Generating offsprings
        while len(solution2) != 2*pop_size:
            a1 = random.randint(0, pop_size-1)
            b1 = random.randint(0, pop_size-1)
            solution2.append(crossover(solution[a1], solution[b1]))
        function1_values2 = [function1(solution2[i])for i in range(0, 2*pop_size)]
        function2_values2 = [function2(solution2[i])for i in range(0, 2*pop_size)]
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
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    # Let's plot the final front now
    function1 = [i * -1 for i in function1_values]
    function2 = [j * -1 for j in function2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
