import math
import time
import os

import numpy as np
import mujoco
import mujoco.viewer
from transforms3d import quaternions
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

class BounceEnv(gym.Env):

    def __init__(self):

        super(BounceEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_space = spaces.Discrete(5)

        # ball_pos_space = spaces.Box(
        #     low=-1, high=1, shape=(3,), dtype=np.float32)
        # board_pos_space = spaces.Box(
        #     low=-1, high=1, shape=(3,), dtype=np.float32)
        # ball_vel_space = spaces.Box(
        #     low=-1, high=1, shape=(3,), dtype=np.float32)
        # # self.observation_space = spaces.Dict({
        #     'ball_velocity': ball_vel_space,
        #     'ball_position': ball_pos_space,
        #     'board_position': board_pos_space,
        # })
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=float)
        print("__init__ called")

    def __del__(self):
        print("__del__ called")

    def step(self, action):
        

        observation = np.array([])
        reward = 0
        done = False
        truncate = False
        info = {}

        return observation, reward, done, truncate, info
    
    def reset(self, seed=None, options=None):

        observation = np.array([])

        # self.reward = 0

        # Implement reset method
        info = {}
        return observation, info