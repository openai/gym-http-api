import gym
import numpy as np
import tensorflow as tf
import random
from abc import ABC, abstractmethod

class Superclass(ABC):
    @abstractmethod
    def print_num(self, num):
        pass

    @abstractmethod
    def print_num_string(self, num):
        print('Printing string of %i' % num, '...')

class Subclass(Superclass):
    def print_num(self, num):
        print(num + 3)

    def print_num_string(self, num):
        super().print_num_string(num)
        for i in range(num):
            print(num)

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
        return obs0, act, rwd[:, np.newaxis], obs1, done[:, np.newaxis]

n = Subclass()
n.print_num(7);
n.print_num_string(7);