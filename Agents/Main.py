
import gym
import numpy as np
import tensorflow as tf
import random
import sys

import DQN
import A2C

if __name__ == '__main__':

	if(len(sys.argv) <= 1):
		print("Need to specify a domain")
		quit()
		
	# Assume command line param will be 'CartPole-v1' or 'MountainCar-v1', etc.
	domain = sys.argv[1]

	env = gym.make(domain) # environment is initialized
	env.seed(1) # seed for environment's random number generator set to 1
	env = env.unwrapped # env retains its characteristics b/c of the declaration but is unwrapped at the same time.

	if(sys.argv[2] == "DQN"):
		agent = DQN.DQN(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[0],
					lr_q_value=0.02, gamma=0.999, epsilon=0.2)
	elif(sys.argv[2] == "A2C"):
		agent = A2C.ActorCritic(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[0],
					lr_actor=0.01, lr_value=0.02, gamma=0.99)
	else:
		print("Invalid algorithm specified. Only DQN and A2C currently supported")
		quit()
				
	nepisode = 1000 # maximum number of episodes = 1000
	iteration = 0 # initialized at the start, or no iterations at beginning

	# Variables pertaining to epsilon decay, which comes later.
	epsilon_step = 10
	epsilon_decay = 0.99
	epsilon_min = 0.001
	
	for i_episode in range(nepisode):
		obs0 = env.reset()
		ep_rwd = 0 # reward initialized to 0; no negative reward, so will always be positive

		while True:

			if i_episode % 10 == 0: env.render()
			act, _ = agent.step(obs0) # the first action will be determined according to obs0.ndim.
	
			obs1, rwd, done, info = env.step(act) # execute this action and observe the resulting reward and image

			agent.memory.store_transition(obs0, act, rwd, obs1, done) # store transition in memory

			obs0 = obs1 # obs0, which will be used in later iterations, becomes the recently manipulated obs1
			ep_rwd += rwd # reward will be gradually compounded on until the loop breaks and a new episode starts

			# The following code can probably be cleaned up in a more general way
			if isinstance(agent, DQN.DQN):
				if iteration >= 128 * 3: # does not occur until three batches have been passed through
					agent.learn() 
					# Epsilon begins to decay every next ten iterations, decreasing the importance of future rewards.
					if iteration % epsilon_step == 0:
						agent.epsilon = max([agent.epsilon * 0.99, 0.001]) # gradient 
	
			if done: # once episode is "done," it will print out its ID and reward and break the while loop.
				# Try to find a cleaner way of doing this
				if isinstance(agent, A2C.ActorCritic):
					_, last_value = agent.step(obs1)
					agent.learn(last_value, done)
					
				print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
				break

			iteration += 1