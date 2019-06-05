#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2ttifrutti, a variant of OpenAI PPO2 baseline.
"""

import tensorflow as tf
import numpy as np
import gym
import gym_remote.exceptions as gre
import os
import math

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import ppo2ttifrutti
import ppo2ttifrutti_policies as policies
import ppo2ttifrutti_sonic_env as env

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2ttifrutti.learn(policy=policies.CnnPolicy,
                            env=SubprocVecEnv([env.make_train_0, env.make_train_1, env.make_train_2, env.make_train_3, env.make_train_4, env.make_train_5, env.make_train_6, env.make_train_7, env.make_train_8, env.make_train_9, env.make_train_10, env.make_train_11, env.make_train_12, env.make_train_13, env.make_train_14, env.make_train_15, env.make_train_16, env.make_train_17, env.make_train_18, env.make_train_19, env.make_train_20, env.make_train_21, env.make_train_22, env.make_train_23, env.make_train_24, env.make_train_25, env.make_train_26, env.make_train_27, env.make_train_28, env.make_train_29, env.make_train_30, env.make_train_31, env.make_train_32, env.make_train_33, env.make_train_34, env.make_train_35, env.make_train_36, env.make_train_37, env.make_train_38, env.make_train_39, env.make_train_40, env.make_train_41, env.make_train_42, env.make_train_43, env.make_train_44, env.make_train_45, env.make_train_46, env.make_val_0, env.make_val_1, env.make_val_2, env.make_val_3, env.make_val_4, env.make_val_5, env.make_val_6, env.make_val_7, env.make_val_8, env.make_val_9, env.make_val_10, env.make_extra_0, env.make_extra_1, env.make_extra_2, env.make_extra_3, env.make_extra_4, env.make_extra_5, env.make_extra_6, env.make_extra_7, env.make_extra_8, env.make_extra_9, env.make_extra_10, env.make_extra_11, env.make_extra_12, env.make_extra_13, env.make_extra_14, env.make_extra_15, env.make_extra_16, env.make_extra_17, env.make_extra_18, env.make_extra_19, env.make_extra_20, env.make_extra_21, env.make_extra_22, env.make_extra_23, env.make_extra_24, env.make_extra_25, env.make_extra_26, env.make_extra_27, env.make_extra_28, env.make_extra_29, env.make_extra_30, env.make_extra_31, env.make_extra_32, env.make_extra_33, env.make_extra_34, env.make_extra_35, env.make_extra_36, env.make_extra_37, env.make_extra_38, env.make_extra_39]),
                            nsteps=2048,
                            nminibatches=16,
                            lam=0.95,
                            gamma=0.99,
                            noptepochs=4,
                            log_interval=1,
                            ent_coef=0.01,
                            lr=lambda _: 2e-4,
                            cliprange=lambda _: 0.1,
                            total_timesteps=int(1e9),
                            save_interval=25)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
