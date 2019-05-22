import gym
import numpy as np
import tensorflow as tf
import random


class Env(object):
    def __init__(self, bits):
        self.bits = bits

    def reset(self):
        self.state = np.random.randint(0, 2, size=self.bits)
        self.goal_state = np.random.randint(0, 2, size=self.bits)
        while (self.state == self.goal_state).all():
            self.goal_state = np.random.randint(0, 2, size=self.bits)
        return self.state, self.goal_state

    def step(self, action):
        self.state = np.copy(self.state)
        self.state[action] = not self.state[action]
        reward = -1.0
        done = False
        if (self.state == self.goal_state).all():
            done = True
            reward = 0.0
        return np.copy(self.state), reward, done


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
            h1 = tf.layers.dense(obs, 256, tf.nn.tanh,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            value = tf.layers.dense(h1, self.act_dim,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            return value

    def get_q_value(self, obs, reuse=False):
        q_value = self.step(obs, reuse)
        return q_value


class DQN(object):
    def __init__(self, act_dim, obs_dim, lr_q_value, gamma, epsilon, tau):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_q_value = lr_q_value
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau

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

        self.q_value_params = tf.global_variables('q_value')
        self.target_q_value_params = tf.global_variables('target_q_value')
        self.target_init_updates = [tf.assign(tq, q) for tq, q in zip(self.target_q_value_params, self.q_value_params)]
        self.target_soft_updates = \
            [tf.assign(tq, (1 - tau) * tq + tau * q)
             for tq, q in zip(self.target_q_value_params, self.q_value_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)

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


bit_size = 15
env = Env(bit_size)

agent = DQN(act_dim=bit_size, obs_dim=bit_size*2,
            lr_q_value=0.001, gamma=0.98, epsilon=0.0, tau=0.05)

nepochs = 10
ncycles = 50
nepisode = 16
nstep = bit_size
noptstep = 40

HER = True
future_k = 4

for i in range(nepochs) :
    for j in range(ncycles) :
        ep_rwd = 0

        for i_episode in range(nepisode):
            obs0, goal = env.reset()
            episode_experience = []

            for t in range(nstep):
                conc_s = np.concatenate([obs0,goal], axis = -1)
                act = agent.step(conc_s)

                obs1, rwd, done = env.step(act)
                episode_experience.append((obs0, act, rwd, obs1, goal, done))

                obs0 = obs1
                ep_rwd += rwd

            for t in range(nstep):
                obs0, act, rwd, obs1, goal, done = episode_experience[t]
                inputs0 = np.concatenate([obs0, goal], axis=-1)
                inputs1 = np.concatenate([obs1, goal], axis=-1)
                agent.memory.store_transition(inputs0, act, rwd, inputs1, done)
                if HER:
                    for h in range(future_k):
                        future = np.random.randint(t, nstep)
                        goal_ = episode_experience[future][3]
                        new_inputs0 = np.concatenate([obs0, goal_], axis=-1)
                        new_inputs1 = np.concatenate([obs1, goal_], axis=-1)

                        if (np.array(obs1) == np.array(goal_)).all():
                            r_ = 0.0
                        else:
                            r_ = -1.0

                        agent.memory.store_transition(new_inputs0, act, r_, new_inputs1, done)

        for t in range(noptstep):
            agent.learn()

        agent.sess.run(agent.target_soft_updates)
        print('after %d cycles, reward is %g' % (i * ncycles + j, ep_rwd))