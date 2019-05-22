import gym
import numpy as np
import tensorflow as tf


class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store_transition(self, obs0, act, rwd):
        self.ep_obs.append(obs0)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)

    def covert_to_array(self):
        array_obs = np.vstack(self.ep_obs)
        array_act = np.array(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []


class ActorNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 10, activation=tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            act_prob = tf.layers.dense(h1, self.act_dim, None,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
        return act_prob

    def choose_action(self, obs, reuse=False):
        act_prob = self.step(obs, reuse)
        softmax_act_prob = tf.nn.softmax(act_prob, name='act_prob')  # use softmax to convert to probability
        return softmax_act_prob

    def get_cross_entropy(self, obs, act, reuse=True):
        act_prob = self.step(obs, reuse)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_prob, labels=act)
        return cross_entropy


class ValueNetwork(object):
    def __init__(self, name):
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 10, activation=tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            value = tf.layers.dense(h1, 1, None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return value

    def get_value(self, obs, reuse=False):
        value = self.step(obs, reuse)
        return value


class ActorCritic:
    def __init__(self, act_dim, obs_dim, lr_actor, lr_value, gamma):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma

        self.OBS = tf.placeholder(tf.float32, [None, self.obs_dim], name="observation")
        self.ACT = tf.placeholder(tf.int32, [None], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")

        actor = ActorNetwork(self.act_dim, 'actor')
        critic = ValueNetwork('critic')
        self.memory = Memory()

        self.act = actor.choose_action(self.OBS)
        cross_entropy = actor.get_cross_entropy(self.OBS, self.ACT)
        self.value = critic.get_value(self.OBS)
        self.advantage = self.Q_VAL - self.value

        actor_loss = tf.reduce_mean(cross_entropy * self.advantage)
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(actor_loss)

        value_loss = tf.reduce_mean(tf.square(self.advantage))
        self.value_train_op = tf.train.AdamOptimizer(self.lr_value).minimize(value_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        prob_weights = self.sess.run(self.act, feed_dict={self.OBS: obs})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        value = self.sess.run(self.value, feed_dict={self.OBS: obs})
        return action, value

    def learn(self, last_value, done):
        obs, act, rwd = self.memory.covert_to_array()

        q_value = self.compute_q_value(last_value, done, rwd)

        self.sess.run(self.actor_train_op, {self.OBS: obs, self.ACT: act, self.Q_VAL: q_value})
        self.sess.run(self.value_train_op, {self.OBS: obs, self.Q_VAL: q_value})

        self.memory.reset()

    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * self.gamma + rwd[t]
            q_value[t] = value
        return q_value[:, np.newaxis]


env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

agent = ActorCritic(act_dim=env.action_space.n, obs_dim=env.observation_space.shape[0],
                    lr_actor=0.01, lr_value=0.02, gamma=0.99)

nepisode = 1000
nstep = 200

for i_episode in range(nepisode):
    obs0 = env.reset()
    ep_rwd = 0

    while True:

        if i_episode % 10 == 0: env.render()
        act, _ = agent.step(obs0)

        obs1, rwd, done, info = env.step(act)

        agent.memory.store_transition(obs0, act, rwd)
        ep_rwd += rwd

        obs0 = obs1

        if done:
            _, last_value = agent.step(obs1)
            agent.learn(last_value, done)
            break

    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)