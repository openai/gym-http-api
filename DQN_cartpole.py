import gym
import numpy as np
import tensorflow as tf
import random


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.index = 0

    def store_transition(self, obs0, act, rwd, obs1, done):
        data = (obs0, act, rwd, obs1, done)
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs0, act, rwd, obs1, done = map(np.stack, zip(*batch))
        return obs0, act, rwd, obs1, done


class QValueNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 10, tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            value = tf.layers.dense(h1, self.act_dim,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return value

    def get_q_value(self, obs, reuse=False):
        q_value = self.step(obs, reuse)
        return q_value


class DQN(object):
    def __init__(self, act_dim, obs_dim, lr_q_value, gamma, epsilon):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_q_value = lr_q_value
        self.gamma = gamma
        self.epsilon = epsilon

        self.OBS0 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations0")
        self.OBS1 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations1")
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.RWD = tf.placeholder(tf.float32, [None], name="reward")
        self.TARGET_Q = tf.placeholder(tf.float32, [None], name="target_q_value")
        self.DONE = tf.placeholder(tf.float32, [None], name="done")

        q_value = QValueNetwork(self.act_dim, 'q_value')
        target_q_value = QValueNetwork(self.act_dim, 'target_q_value')
        self.memory = ReplayBuffer(capacity=int(1e6))

        self.q_value0 = q_value.get_q_value(self.OBS0)

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
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(action, axis=1)[0]
        return action

    def learn(self):
        obs0, act, rwd, obs1, done = self.memory.sample(batch_size=128)

        target_q_value1 = self.sess.run(self.target_q_value1,
                                        feed_dict={self.OBS1: obs1, self.RWD: rwd, self.DONE: np.float32(done)})

        self.sess.run(self.q_value_train_op,feed_dict={self.OBS0: obs0, self.ACT: act,
                                                       self.TARGET_Q: target_q_value1})

        self.sess.run(self.target_updates)


env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped


agent = DQN(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[0],
            lr_q_value=0.02, gamma=0.999, epsilon=0.2)

nepisode = 1000
iteration = 0

epsilon_step = 10
epsilon_decay = 0.99
epsilon_min = 0.001


for i_episode in range(nepisode):
    obs0 = env.reset()
    ep_rwd = 0

    while True:
        env.render()
        act = agent.step(obs0)

        obs1, rwd, done, _ = env.step(act)

        agent.memory.store_transition(obs0, act, rwd, obs1, done)

        obs0 = obs1
        ep_rwd += rwd

        if iteration >= 128 * 3:
            agent.learn()
            if iteration % epsilon_step == 0:
                agent.epsilon = max([agent.epsilon * 0.99, 0.001])

        if done:
            print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
            break

        iteration += 1