#!/usr/bin/env python

import numpy as np
import gym
import gym_remote.client as grc
from retro_contest.local import make
from baselines.common.atari_wrappers import FrameStack

import cv2
cv2.ocl.setUseOpenCL(False)

class WarpFrame96(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 96x96."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

def make_custom(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    #env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame96(env)
    if stack:
        env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    return env

def make_train(env_idx, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    dicts = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'}, #0
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'}, #5---
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}, #12
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'EmeraldHillZone.Act1'}, #13
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'EmeraldHillZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'ChemicalPlantZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'ChemicalPlantZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MetropolisZone.Act2'}, #18---
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'OilOceanZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'OilOceanZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MysticCaveZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'MysticCaveZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'HillTopZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'CasinoNightZone.Act1'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'WingFortressZone'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'AquaticRuinZone.Act2'},
        {'game': 'SonicTheHedgehog2-Genesis', 'state': 'AquaticRuinZone.Act1'}, #27
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LavaReefZone.Act2'}, #28
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MarbleGardenZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MarbleGardenZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'DeathEggZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'DeathEggZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'FlyingBatteryZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'SandopolisZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'SandopolisZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'HiddenPalaceZone'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'HydrocityZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'IcecapZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'IcecapZone.Act2'}, #43---
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'AngelIslandZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LaunchBaseZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LaunchBaseZone.Act1'} #46
    ]
    print(dicts[env_idx]['game'], dicts[env_idx]['state'], flush=True)
    env = make(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'])#, bk2dir='/tmp')#, record='/tmp')
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame96(env)
    if stack:
        env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    return env

def make_train_0(stack=True, scale_rew=True):
    return make_train(0, stack, scale_rew)

def make_train_1(stack=True, scale_rew=True):
    return make_train(1, stack, scale_rew)

def make_train_2(stack=True, scale_rew=True):
    return make_train(2, stack, scale_rew)

def make_train_3(stack=True, scale_rew=True):
    return make_train(3, stack, scale_rew)

def make_train_4(stack=True, scale_rew=True):
    return make_train(4, stack, scale_rew)

def make_train_5(stack=True, scale_rew=True):
    return make_train(5, stack, scale_rew)

def make_train_6(stack=True, scale_rew=True):
    return make_train(6, stack, scale_rew)

def make_train_7(stack=True, scale_rew=True):
    return make_train(7, stack, scale_rew)

def make_train_8(stack=True, scale_rew=True):
    return make_train(8, stack, scale_rew)

def make_train_9(stack=True, scale_rew=True):
    return make_train(9, stack, scale_rew)

def make_train_10(stack=True, scale_rew=True):
    return make_train(10, stack, scale_rew)

def make_train_11(stack=True, scale_rew=True):
    return make_train(11, stack, scale_rew)

def make_train_12(stack=True, scale_rew=True):
    return make_train(12, stack, scale_rew)

def make_train_13(stack=True, scale_rew=True):
    return make_train(13, stack, scale_rew)

def make_train_14(stack=True, scale_rew=True):
    return make_train(14, stack, scale_rew)

def make_train_15(stack=True, scale_rew=True):
    return make_train(15, stack, scale_rew)

def make_train_16(stack=True, scale_rew=True):
    return make_train(16, stack, scale_rew)

def make_train_17(stack=True, scale_rew=True):
    return make_train(17, stack, scale_rew)

def make_train_18(stack=True, scale_rew=True):
    return make_train(18, stack, scale_rew)

def make_train_19(stack=True, scale_rew=True):
    return make_train(19, stack, scale_rew)

def make_train_20(stack=True, scale_rew=True):
    return make_train(20, stack, scale_rew)

def make_train_21(stack=True, scale_rew=True):
    return make_train(21, stack, scale_rew)

def make_train_22(stack=True, scale_rew=True):
    return make_train(22, stack, scale_rew)

def make_train_23(stack=True, scale_rew=True):
    return make_train(23, stack, scale_rew)

def make_train_24(stack=True, scale_rew=True):
    return make_train(24, stack, scale_rew)

def make_train_25(stack=True, scale_rew=True):
    return make_train(25, stack, scale_rew)

def make_train_26(stack=True, scale_rew=True):
    return make_train(26, stack, scale_rew)

def make_train_27(stack=True, scale_rew=True):
    return make_train(27, stack, scale_rew)

def make_train_28(stack=True, scale_rew=True):
    return make_train(28, stack, scale_rew)

def make_train_29(stack=True, scale_rew=True):
    return make_train(29, stack, scale_rew)

def make_train_30(stack=True, scale_rew=True):
    return make_train(30, stack, scale_rew)

def make_train_31(stack=True, scale_rew=True):
    return make_train(31, stack, scale_rew)

def make_train_32(stack=True, scale_rew=True):
    return make_train(32, stack, scale_rew)

def make_train_33(stack=True, scale_rew=True):
    return make_train(33, stack, scale_rew)

def make_train_34(stack=True, scale_rew=True):
    return make_train(34, stack, scale_rew)

def make_train_35(stack=True, scale_rew=True):
    return make_train(35, stack, scale_rew)

def make_train_36(stack=True, scale_rew=True):
    return make_train(36, stack, scale_rew)

def make_train_37(stack=True, scale_rew=True):
    return make_train(37, stack, scale_rew)

def make_train_38(stack=True, scale_rew=True):
    return make_train(38, stack, scale_rew)

def make_train_39(stack=True, scale_rew=True):
    return make_train(39, stack, scale_rew)

def make_train_40(stack=True, scale_rew=True):
    return make_train(40, stack, scale_rew)

def make_train_41(stack=True, scale_rew=True):
    return make_train(41, stack, scale_rew)

def make_train_42(stack=True, scale_rew=True):
    return make_train(42, stack, scale_rew)

def make_train_43(stack=True, scale_rew=True):
    return make_train(43, stack, scale_rew)

def make_train_44(stack=True, scale_rew=True):
    return make_train(44, stack, scale_rew)

def make_train_45(stack=True, scale_rew=True):
    return make_train(45, stack, scale_rew)

def make_train_46(stack=True, scale_rew=True):
    return make_train(46, stack, scale_rew)