import gym
import numpy as np
import tensorflow as tf
import random
import sys

import DDPG

if __name__ == '__main__':

	if(len(sys.argv) <= 1):
		print("Need to specify a domain")
		quit()
		
	# Assume command line param will be 'CartPole-v1' or 'MountainCar-v1', etc.
	domain = sys.argv[1]

	env = gym.make(domain)
	env.seed(1)
	env = env.unwrapped

	if(sys.argv[2] == "DDPG"):
		agent = DDPG.DDPG(act_dim=env.action_space.shape[0], obs_dim=env.observation_space.shape[0],
					lr_actor=0.0001, lr_q_value=0.001, gamma=0.99, tau=0.01, action_noise_std=1)

	else:
		print("Invalid algorithm specified. Only DDPG currently supported")
		quit()

	nepisode = 1000
	nstep = 200
	iteration = 0


	for i_episode in range(nepisode):
		obs0 = env.reset()
		ep_rwd = 0

		for t in range(nstep):
			if i_episode % 10 == 0: env.render()
			act = agent.step(obs0)

			obs1, rwd, done, _ = env.step(act)

			agent.memory.store_transition(obs0, act, rwd/10, obs1, done)

			obs0 = obs1
			ep_rwd += rwd

			if iteration >= 128 * 3:
				agent.learn()

			iteration += 1

		print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)