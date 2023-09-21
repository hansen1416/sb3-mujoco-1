import os
import time
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from train import train_agent


class CustomHumanoidEnv(HumanoidEnv):
    def __init__(self, render_mode='human', **kwargs):
        super().__init__(**kwargs)
        self.render_mode = render_mode

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            self.render_mode = mode
        return super().render(**kwargs)


class CustomHumanoidStandupEnv(HumanoidStandupEnv):
    def __init__(self, render_mode='rgb_array', **kwargs):
        super().__init__(**kwargs)
        self.render_mode = render_mode

    def render(self, mode='rgb_array', **kwargs):
        if mode == 'human':
            self.render_mode = mode
        return super().render(**kwargs)


def demo():

    # env = gym.make('Humanoid-v4')
    env = CustomHumanoidEnv()

    obs = env.reset()

    env.render_mode = "human"

    model = PPO('MlpPolicy', env, verbose=1)

    for i in range(1000):
        # action, _states = model.predict(obs, deterministic=True)
        action = model.action_space.sample()
        # print(f"action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


def test():

    model = PPO.load(
        "models/CustomHumanoidStandupEnv-PPO/1500000.zip")

    # print(model)
    env = CustomHumanoidStandupEnv(render_mode="human")
    obs, _ = env.reset()

    # print(obs)
    while True:
        action, _ = model.predict(obs)

        obs, rewards, dones, truncate, info = env.step(action)

        env.render()

        print("action: {}, reward: {}".format(action, rewards))


if __name__ == "__main__":
    # demo()

    # train_agent(CustomHumanoidStandupEnv(), algorithm=PPO)

    test()
