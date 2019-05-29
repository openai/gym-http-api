import gym
import numpy as np
import tensorflow as tf
import random
import sys

import A2C
import DQN
import PG

import gym_cmd
parser = build_parser()
options = parser.parse_args()

if __name__ == '__main__':

	# Assume command line param will be 'CartPole-v1' or 'MountainCar-v1', etc.
	domain = options.domain

	env = gym.make(domain)
	env.seed(1)
	env = env.unwrapped
ys.argv[
	if(s2] == "DQN"):
		agent = DQN.DQN(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[0],
					lr_q_value=options.q_val, gamma=options.gamma, epsilon=options.epsilon)
	elif(sys.argv[2] == "A2C"):
		agent = A2C.ActorCritic(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[0],
					lr_actor=0.01, lr_value=0.02, gamma=0.99)
	elif(sys.argv[2] == "PG"):
		agent = PG.PolicyGradient(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[0], 
					lr=0.02, gamma=0.99)

	else:
		print("Invalid algorithm specified. Only A2C, DQN, and PG currently supported")
		quit()
				
	nepisode = 1000
	batch_size = 128
	iteration = 0
	nstep = 200
	
	# Necessary for DQN
	epsilon_step = 10
	epsilon_decay = 0.99
	epsilon_min = 0.001
	
	for i_episode in range(nepisode):
		obs0 = env.reset()
		ep_rwd = 0

		# Discrete domains
		while True:

			if i_episode % 10 == 0: env.render()
			act, _ = agent.step(obs0)
	
			obs1, rwd, done, info = env.step(act)

			agent.memory.store_transition(obs0, act, rwd, obs1, done)

			obs0 = obs1
			ep_rwd += rwd

			# The following code can probably be cleaned up in a more general way
			if isinstance(agent, DQN.DQN):
				if iteration >= 128 * 3:
					agent.learn() 
					if iteration % epsilon_step == 0:
						agent.epsilon = max([agent.epsilon * 0.99, 0.001])
	
			if done:
				# Try to find a cleaner way of doing this
				if isinstance(agent, A2C.ActorCritic):
					_, last_value = agent.step(obs1)
					agent.learn(last_value, done)
				if isinstance(agent, PG.PolicyGradient):
					agent.learn()

				print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
				break

			iteration += 1