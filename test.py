from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, DQN

from mujoco_xml import arm_xml
from PunchEnv import PunchEnv


# env = BounceEnv(ws_connection=ws)
# env.reset()

model = PPO.load("models/punch-ppo/100000.zip")

# print(model)
env = PunchEnv()
obs, _ = env.reset()


if __name__ == "__main__":

    # print(obs)
    while True:
        action, _ = model.predict(obs)

        obs, rewards, dones, truncate, info = env.step(action)

        env.render()

        print("action: {}, reward: {}".format(action, rewards))
