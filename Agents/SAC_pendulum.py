import numpy as np
import tensorflow as tf
import gym
import random

EPS = 1e-8

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.index = 0

    def store_transition(self, obs0, act, rwd, obs1, done):
        data = (obs0, act, rwd, obs1, done)
        if self.capacity >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs0, act, rwd, obs1, done = map(np.stack, zip(*batch))
        return obs0, act, rwd, obs1, done


class ValueNetwork(object):
    def __init__(self, name):
        self.name = name

    def step(self, obs):
        with tf.variable_scope(self.name):
            h1 = tf.layers.dense(obs, 300, tf.nn.relu)
            value = tf.layers.dense(h1, 1)
            value = tf.squeeze(value, axis=1)
            return value

    def get_value(self, obs):
        value = self.step(obs)
        return value


class QValueNetwork(object):
    def __init__(self, name):
        self.name = name

    def step(self, obs, action, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            input = tf.concat([obs, action], axis=-1)
            h1 = tf.layers.dense(input, 300, tf.nn.relu)
            q_value = tf.layers.dense(h1, 1)
            q_value = tf.squeeze(q_value, axis=1)
            return q_value

    def get_q_value(self, obs, action, reuse=False):
        q_value = self.step(obs, action, reuse)
        return q_value


class ActorNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim
        self.name = name

    def step(self, obs, log_std_min=-20, log_std_max=2):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(obs, 300, tf.nn.relu)
            h2 = tf.layers.dense(h1, 300, tf.nn.relu)
            mu = tf.layers.dense(h2, self.act_dim, None)
            log_std = tf.layers.dense(h2, self.act_dim, tf.tanh)
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std

            pre_sum = -0.5 * (((pi - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            logp_pi = tf.reduce_sum(pre_sum, axis=1)

            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

            clip_pi = 1 - tf.square(pi)
            clip_up = tf.cast(clip_pi > 1, tf.float32)
            clip_low = tf.cast(clip_pi < 0, tf.float32)
            clip_pi = clip_pi + tf.stop_gradient((1 - clip_pi) * clip_up + (0 - clip_pi) * clip_low)

            logp_pi -= tf.reduce_sum(tf.log(clip_pi + 1e-6), axis=1)
        return mu, pi, logp_pi

    def evaluate(self, obs):
        mu, pi, logp_pi = self.step(obs)
        action_scale = 2.0 # env.action_space.high[0]
        mu *= action_scale
        pi *= action_scale
        return mu, pi, logp_pi


class SAC(object):
    def __init__(self, act_dim, obs_dim, lr_actor, lr_value, gamma, tau, alpha=0.2):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma
        self.tau = tau

        self.OBS0 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations0")
        self.OBS1 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations1")
        self.ACT = tf.placeholder(tf.float32, [None, self.act_dim], name="action")
        self.RWD = tf.placeholder(tf.float32, [None,], name="reward")
        self.DONE = tf.placeholder(tf.float32, [None,], name="done")

        policy = ActorNetwork(self.act_dim, 'Actor')
        q_value_net_1 = QValueNetwork('Q_value1')
        q_value_net_2 = QValueNetwork('Q_value2')
        value_net = ValueNetwork('Value')
        target_value_net = ValueNetwork('Target_Value')

        self.replay_buffer = ReplayBuffer(capacity=int(1e6))

        mu, self.pi, logp_pi = policy.evaluate(self.OBS0)

        q_value1 = q_value_net_1.get_q_value(self.OBS0, self.ACT)
        q_value1_pi = q_value_net_1.get_q_value(self.OBS0, self.pi, reuse=True)
        q_value2 = q_value_net_2.get_q_value(self.OBS0, self.ACT)
        q_value2_pi = q_value_net_2.get_q_value(self.OBS0, self.pi, reuse=True)
        value = value_net.get_value(self.OBS0)
        target_value = target_value_net.get_value(self.OBS1)

        min_q_value_pi = tf.minimum(q_value1_pi, q_value2_pi)
        next_q_value = tf.stop_gradient(self.RWD + self.gamma * (1 - self.DONE) * target_value)
        next_value = tf.stop_gradient(min_q_value_pi - alpha * logp_pi)

        policy_loss = tf.reduce_mean(alpha * logp_pi - q_value1_pi)
        q_value1_loss = tf.reduce_mean(tf.squared_difference(next_q_value, q_value1))
        q_value2_loss = tf.reduce_mean(tf.squared_difference(next_q_value, q_value2))
        value_loss = tf.reduce_mean(tf.squared_difference(next_value, value))
        total_value_loss = q_value1_loss + q_value2_loss + value_loss

        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_actor)
        actor_train_op = actor_optimizer.minimize(policy_loss, var_list=tf.global_variables('Actor'))
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_value)
        value_params = tf.global_variables('Q_value') + tf.global_variables('Value')

        with tf.control_dependencies([actor_train_op]):
            value_train_op = value_optimizer.minimize(total_value_loss, var_list=value_params)
        with tf.control_dependencies([value_train_op]):
            self.target_update = [tf.assign(tv, self.tau * tv + (1 - self.tau) * v)
                             for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        target_init = [tf.assign(tv, v)
                       for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

    def step(self, obs):
        action = self.sess.run(self.pi, feed_dict={self.OBS0: obs.reshape(1, -1)})
        action = np.squeeze(action, axis=1)
        print(action)
        return action

    def learn(self):
        obs0, act, rwd, obs1, done = self.replay_buffer.sample(batch_size=128)
        feed_dict = {self.OBS0: obs0, self.ACT: act,self.OBS1: obs1, self.RWD: rwd,
                                                               self.DONE: np.float32(done)}
        self.sess.run(self.target_update, feed_dict)

env = gym.make("Pendulum-v0")
env.seed(1)
env = env.unwrapped

agent = SAC(act_dim=env.action_space.shape[0], obs_dim=env.observation_space.shape[0],
            lr_actor=1e-3, lr_value=1e-3, gamma=0.99, tau=0.995)

nepisode = 1000
nstep = 200
batch_size = 128
iteration = 0

for i_episode in range(nepisode):
    obs0 = env.reset()

    ep_rwd = 0

    for t in range(nstep):
        if i_episode % 10 == 0: env.render()
        act = agent.step(obs0)

        obs1, rwd, done, _ = env.step(act)

        agent.replay_buffer.store_transition(obs0, act, rwd/10, obs1, done)

        obs0 = obs1
        ep_rwd += rwd

        if iteration >= batch_size * 3:
            agent.learn()

        iteration += 1

    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
