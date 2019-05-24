""" This file will import four packages: gym (the OpenAI Gym package for developing and comparing 
reinforcement learning algorithms), numpy (a package used primarily to conduct complex mathematical operations
and to organize stacks along new axes), tensorflow (the computational library used most considerably in machine 
learning), and random (a package used simply to generate random numbers) 
"""
import gym
import numpy as np
import tensorflow as tf
import random

""" A replay buffer exists due to the nature of DRL being conducted through Python applications. Normally, a
buffer of a fixed capacity is created to learn from a certain number of episodes, and it flushes out episodes
once it stores that number. However, this implementation simply wraps around and replaces episodes with new ones.
"""
class ReplayBuffer(object):
    def __init__(self, capacity): # empty buffer of size "capacity" is initialized; index points to start.
        self.buffer = []
        self.capacity = capacity
        self.index = 0

    def store_transition(self, obs0, act, rwd, obs1, done):
        data = (obs0, act, rwd, obs1, done)
        if self.index >= len(self.buffer): # if the index is greater than the length of buffer, append to buffer
            self.buffer.append(data)
        else: # otherwise, data becomes stored at the index
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.capacity # used to wrap around if index goes over capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # a batch becomes any *batch_size* episodes in the buffer
        # The batch puts out its observation spaces, action space, reward, and "done" values
        obs0, act, rwd, obs1, done = map(np.stack, zip(*batch))
        return obs0, act, rwd[:, np.newaxis], obs1, done[:, np.newaxis]

# The next class will represent the network's architecture.
class QValueNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim # network's action space
        self.name = name # network's name

    def step(self, obs, reuse): # step will receive a Q-value according to two layers: h1 and value.
        with tf.variable_scope(self.name, reuse=reuse):
            # recall that the Q-values will be calculated according to random weights:
            h1 = tf.layers.dense(obs, 10, tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            value = tf.layers.dense(h1, self.act_dim,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return value

    def get_q_value(self, obs, reuse=False): # access method to receive a Q-value
        q_value = self.step(obs, reuse)
        return q_value

# Finally, we have a class dedicated to the network's specifications.
class DQN(object):
    def __init__(self, act_dim, obs_dim, lr_q_value, gamma, epsilon):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_q_value = lr_q_value
        self.gamma = gamma # 'discount factor' between 0 and 1 that states how important future rewards are
        self.epsilon = epsilon # the percentage (between 0 and 1) of how many of our actions are random

        # These variables are declared as Tensorflow "placeholders" and will be assigned values later.
        # Additionally, we initialize two observation spaces in order to properly "reset" environments.
        self.OBS0 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations0")
        self.OBS1 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations1")
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.RWD = tf.placeholder(tf.float32, [None], name="reward")
        self.TARGET_Q = tf.placeholder(tf.float32, [None], name="target_q_value")
        self.DONE = tf.placeholder(tf.float32, [None], name="done")

        q_value = QValueNetwork(self.act_dim, 'q_value') # one network is generated to receive Q-value
        target_q_value = QValueNetwork(self.act_dim, 'target_q_value') # another generated to receive target
        self.memory = ReplayBuffer(capacity=int(1e6)) # system memory is initialized to capacity of 1 million.

        self.q_value0 = q_value.get_q_value(self.OBS0) # receives Q-value from first network

        self.action_onehot = tf.one_hot(self.ACT, self.act_dim, dtype=tf.float32)
        self.q_value_onehot = tf.reduce_sum(tf.multiply(self.q_value0, self.action_onehot), axis=1)

        self.target_q_value1 = self.RWD + (1. - self.DONE) * self.gamma \
                               * tf.reduce_max(target_q_value.get_q_value(self.OBS1), axis=1)

        q_value_loss = tf.reduce_mean(tf.square(self.q_value_onehot - self.TARGET_Q))
        self.q_value_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_q_value).minimize(q_value_loss)

        # Jacob: Why doesn't this work? global_variables should accept a scope parameter
        #self.q_value_params = tf.global_variables('q_value')
        self.q_value_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_value')
        # Jacob: Why doesn't this work? global_variables should accept a scope parameter
        #self.target_q_value_params = tf.global_variables('target_q_value')
        self.target_q_value_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_value')
        self.target_updates = [tf.assign(tq, q) for tq, q in zip(self.target_q_value_params, self.q_value_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_updates)

    def step(self, obs):
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        action = self.sess.run(self.q_value0, feed_dict={self.OBS0: obs})
        # if our random value is less than epsilon, take the "exploration" approach - next action will be random.
        if np.random.rand(1) < self.epsilon:
            action = np.random.normal(0, 1)
            action = np.clip(action, -2, 2)
        # otherwise, we take the "exploitation" approach and apply the best action at the next step.
        else:
            action = np.argmax(action, axis=1)[0]
        action_arr = [action]
        return action_arr

    def learn(self): # will learn according to a given batch.
        obs0, act, rwd, obs1, done = self.memory.sample(batch_size=128)

        target_q_value1 = self.sess.run(self.target_q_value1,
                                        feed_dict={self.OBS1: obs1, self.RWD: rwd, self.DONE: np.float32(done)})

        self.sess.run(self.q_value_train_op,feed_dict={self.OBS0: obs0, self.ACT: act,
                                                       self.TARGET_Q: target_q_value1})

        self.sess.run(self.target_updates)


env = gym.make('Pendulum-v0') # environment is initialized according to CartPole specifications
env.seed(1) # seed for environment's random number generator set to 1
env = env.unwrapped # env retains its characteristics b/c of the declaration but is unwrapped at the same time.

# DQN is initialized such that it takes on specific values for its Q, gamma, and epsilon.
agent = DQN(act_dim=env.action_space.shape[0], obs_dim=env.observation_space.shape[0],
            lr_q_value=0.02, gamma=0.999, epsilon=0.2)

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

        print(agent.step(obs0))
        if i_episode % 10 == 0: env.render()
        act = agent.step(obs0) # the first action will be determined according to obs0.ndim.

        obs1, rwd, done, _ = env.step(act) # execute this action and observe the resulting reward and image

        agent.memory.store_transition(obs0, act, rwd/10, obs1, done) # store transition in memory

        obs0 = obs1 # obs0, which will be used in later iterations, becomes the recently manipulated obs1
        ep_rwd += rwd # reward will be gradually compounded on until the loop breaks and a new episode starts

        print(ep_rwd)
        if iteration >= 128 * 3: # does not occur until three batches have been passed through
            agent.learn() 
            # Epsilon begins to decay every next ten iterations, decreasing the importance of future rewards.
            if iteration % epsilon_step == 0:
                agent.epsilon = max([agent.epsilon * 0.99, 0.001]) # gradient 

        if done: # once episode is "done," it will print out its ID and reward and break the while loop.
            print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
            break

        iteration += 1