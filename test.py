from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import numpy as np

from mujoco_env import MujocoEnv
from mujoco_xml import arm_xml


class myEnv(MujocoEnv):
    def __init__(self, model_path,
                 frame_skip,
                 observation_space,):

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 500,
        }

        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode="human"
        )

    def reset_model(self):
        super()._reset_simulation()

        observation = np.zeros(20, dtype=np.float32)

        return observation

    def step(self, action):

        observation = np.zeros(20, dtype=np.float32)
        reward = 0
        done = False
        truncate = False
        info = {}

        return observation, reward, done, truncate, info


if __name__ == "__main__":

    env = myEnv(arm_xml, 1, spaces.Box(
        low=-1.0, high=1.0, shape=(20,), dtype=float),

    )

    # check_env(env)

    while True:
        env.render()
