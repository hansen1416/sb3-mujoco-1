"""
Tips:

RL, data is collected through interactions with the environment by the agent itself, 
this could lead to vicious circle, so RL may vary from one run to another. 
always do several runs to have quantitative results.

RL are generally dependent on finding appropriate hyperparameters.
A best practice when you apply RL to a new problem is to do automatic hyperparameter optimization. 

When applying RL to a custom problem, you should always normalize the input to the agent,
and look at common preprocessing done on other environments

This reward engineering, necessitates several iterations,  
Deep Mimic combines imitation learning and reinforcement learning to do acrobatic moves.

RL is the instability of training. You can observe during training a huge drop in performance. 
This behavior is particularly present in DDPG, that's why its extension TD3 tries to tackle that issue. 
Other method, like TRPO or PPO make use of a trust region to minimize that problem by avoiding too large update.

Because most algorithms use exploration noise during training, 
you need a separate test environment to evaluate the performance of your agent at a given time. 
It is recommended to periodically evaluate your agent for n test episodes (n is usually between 5 and 20) 
and average the reward per episode to have a good estimate.
"""

import math

import numpy as np
import mujoco
import mujoco.viewer
from transforms3d import quaternions
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from xml import arm_xml


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction


class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class PunchEnv(gym.Env):

    def __init__(self):

        super(PunchEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=float)

        # todo this is linear speed, add acceleration
        self.shoulder_angle = np.linspace(10, 86, 100)
        self.elbow_angle = np.linspace(110, 0, 100)
        self.motion_idx = 0
        self.direction = 1

        self.model = mujoco.MjModel.from_xml_string(arm_xml)
        self.data = mujoco.MjData(self.model)

        self.previous_contact_state = False
        self.current_contact_state = False
        self.ncon = 0

        print("__init__ called")

    def __del__(self):
        print("__del__ called")

    def step(self, action):

        # print(self.data.qpos)
        # print(self.data.qvel)

        mujoco.mj_step(self.model, self.data)

        # joint manipulation start
        q = quaternions.axangle2quat(
            [0, 1, 0], math.radians(self.shoulder_angle[self.motion_idx]), is_normalized=True)

        self.data.qpos[0] = q[0]  # w
        self.data.qpos[1] = q[1]  # x
        self.data.qpos[2] = q[2]  # y
        self.data.qpos[3] = q[3]  # z

        self.data.qpos[4] = math.radians(
            self.elbow_angle[self.motion_idx])  # w
        self.data.qvel[3] = 0
        # print(data.qvel)

        if action == 1:
            self.motion_idx += 1
        elif action == 0:
            self.motion_idx -= 1

        if self.motion_idx >= len(self.elbow_angle) - 1:
            self.motion_idx = len(self.elbow_angle) - 1
        elif self.motion_idx <= 0:
            self.motion_idx = 0
        # joint manipulation end

        # assemble observation
        # cartesian position of upper arm, lower arm, hand and ball
        # quaternion of upper arm, lower arm
        upper_arm_p = normalize(
            self.data.geom_xpos[self.model.geom('upper_arm').id])
        lower_arm_p = normalize(
            self.data.geom_xpos[self.model.geom('lower_arm').id])
        hand_p = normalize(self.data.geom_xpos[self.model.geom('hand').id])
        ball_p = normalize(self.data.geom_xpos[self.model.geom('ball').id])

        upper_arm_r = normalize(quaternions.mat2quat(
            self.data.geom_xmat[self.model.geom('upper_arm').id]))
        lower_arm_r = normalize(quaternions.mat2quat(
            self.data.geom_xmat[self.model.geom('lower_arm').id]))

        # concatenate numpy arrays
        observation = np.concatenate(
            (upper_arm_p, lower_arm_p, hand_p, ball_p, upper_arm_r, lower_arm_r), axis=None)

        # print('--------------- observation')
        # print(observation)

        # reward is

        # print('--------------- acc')
        # acceleration is of shape (10,) 3 for shoulder ball joint, 1 for elbow hinge joint, 6 for sphere free joint
        reward = np.sum(self.data.qacc[4:])

        if self.data.ncon > 0:
            self.current_contact_state = True

        if self.previous_contact_state == False and self.current_contact_state == True:
            self.ncon += 1

        self.previous_contact_state = self.current_contact_state

        # when contact twice, done
        if self.ncon >= 2:
            done = True
        else:
            done = False

        truncate = False
        info = {}

        return observation, reward, done, truncate, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)

        self.previous_contact_state = False
        self.current_contact_state = False
        self.ncon = 0

        observation = np.zeros(20, dtype=np.float32)

        self.reward = 0

        # Implement reset method
        info = {}
        return observation, info


if __name__ == "__main__":

    env = PunchEnv()

    check_env(env)
