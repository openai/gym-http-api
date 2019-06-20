# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Copied from Training.py in the sonicNEAT repo
import retro
import numpy as np
import cv2
import neat
import pickle
from platform import dist


#First function to optimize
def function1(x):
    value = -x**2
    return value

#Second function to optimize
def function2(x):
    value = -(x-2)**2
    return value

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def calculate_novelty(behavior_characterizations):
    ns = []
    for i in range(0, len(behavior_characterizations)):
        total_dist = 0
        for j in range(0, len(behavior_characterizations)):
            if(i != j):
                i_behavior = behavior_characterizations[i]
                j_behavior = behavior_characterizations[j]
                dist = euclidean_dist(i_behavior, j_behavior)
                total_dist += dist
        avg_dist = total_dist / (len(solution) - 1)
        ns.append(avg_dist)
    return ns

def euclidean_dist(i_behavior, j_behavior):
    index = total = 0
    minimum = min(len(i_behavior), len(j_behavior))
    while index < minimum: 
        total += (i_behavior[index] - j_behavior[index]) ** 2
        index += 1
    while index < len(i_behavior):
        total += i_behavior[index] ** 2
        index += 1
    while index < len(j_behavior):
        total += j_behavior[index] ** 2
        index += 1
    total = math.sqrt(total)
    return total
    
#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution

# Copied from Training.py in the sonicNEAT repo
def evaluate(env,net):
    ob = env.reset()
    ac = env.action_space.sample()
    inx, iny, inc = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)
    current_max_fitness = 0
    fitness_current = 0
    frame = 0
    counter = 0
    xpos = 0
    done = False
    behaviors = []
            
    while not done:
        
        env.render()
        frame += 1
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx,iny))

        imgarray = np.ndarray.flatten(ob)

        nnOutput = net.activate(imgarray)
            
        ob, rew, done, info = env.step(nnOutput)
            
        xpos = info['x']
        ypos = info['y']
        behaviors.append(xpos)
        behaviors.append(ypos)
            
        if xpos >= 65664:
                fitness_current += 10000000
                done = True
            
        fitness_current += rew
            
        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            counter = 0
        else:
            counter += 1
                
        if done or counter == 250:
            done = True
            #print(fitness_current)
    
    # Add code to return the behavior characterization as well.
    return fitness_current, behaviors

def random_genome(n):
    # n is the number of weights
    return np.random.rand(1,n)
    
if __name__ == '__main__':
    #Main program starts here
    
    # Copied from Training.py in the sonicNEAT repo
    env = retro.make(game = "SonicTheHedgehog-Genesis", state = "GreenHillZone.Act1")
    imgarray = []
    xpos_end = 0
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
    p = neat.Population(config)
    
    # Schrum: Makes these small to test at first
    max_gen = 3

    #Initialization
    # Schrum: This will need to be replaced with initialization for the network weights ... probably from -1 to 1, but how many will you need? Depends on network architecture.
    num_weights = 10 # What should this actually be?
    #solution=[random_genome(num_weights) for i in range(0,pop_size)]
    gen_no=0
    while(gen_no<max_gen):
        fitness_scores = []
        behavior_characterizations = []
        # Copied/Adapted from Training.py in the sonicNEAT repo
        for genome_id in p.population:
            genome = p.population[genome_id]
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            fitness, behavior_char = evaluate(env,net)
            fitness_scores.append(fitness)
            behavior_characterizations.append(behavior_char)
            
        # Schrum: Here is where you have to compare all of the behavior characterizations to get the diversity/novelty scores.
        novelty_scores = calculate_novelty(behavior_characterizations)
        print(novelty_scores)
        
        function1_values = fitness_scores
        function2_values = novelty_scores


        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
        print("The best front for Generation number ",gen_no, " is")
        for valuez in non_dominated_sorted_solution[0]:
            print(round(solution[valuez],3),end=" ")
        print("\n")
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]
        #Generating offsprings
        while(len(solution2)!=2*pop_size):
            a1 = random.randint(0,pop_size-1)
            b1 = random.randint(0,pop_size-1)
            solution2.append(crossover(solution[a1],solution[b1]))
        function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
        function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    #Lets plot the final front now
    function1 = [i * -1 for i in function1_values]
    function2 = [j * -1 for j in function2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
