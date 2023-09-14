import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
import websocket
import json
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

"""
game env is a 3D game, the ball is bouncing in a 3D cuboid with one side be empty, the board is trying to catch the ball
if the ball is bounced out of the cuboid, the game is over.
the board can move in 4 directions, left, right, up, down
the ball is bouncing in a random direction, when the game is over, 
the ball will be reset to the center of the cuboid with a random velocity

when the board catches the ball, the reward is 1
when the ball is bounced out of the cuboid, the reward is -1

"""


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


class BounceEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, ws_connection=None):

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

        self.reward = 0

        self.ws_connection = ws_connection

        print("__init__ called")

    def __del__(self):
        print("__del__ called")

    def step(self, action):

        # send action to the game
        if action == 0:
            self.ws_connection.send('')
        elif action == 1:
            self.ws_connection.send("s")
        elif action == 2:
            self.ws_connection.send("a")
        elif action == 3:
            self.ws_connection.send("d")
        elif action == 4:
            self.ws_connection.send("w")

        try:
            msg = self.ws_connection.recv()
        except websocket.WebSocketTimeoutException:
            print("Timeout occurred")
            return self.observation, self.reward, False, False, {}

        try:
            msg = json.loads(msg)
        except:
            print("Illegal message")
            print(msg)
            return self.observation, self.reward, False, False, {}

        # self.observation: ObservationDict = {"ball_position": np.array([msg_obs[0], msg_obs[1], msg_obs[2]], dtype=np.float32),
        #                                      "board_position": np.array([msg_obs[3], msg_obs[4], msg_obs[5]], dtype=np.float32),
        #                                      "ball_velocity": np.array([msg_obs[6], msg_obs[7], msg_obs[8]], dtype=np.float32)}

        self.observation = np.array(msg['observation'], dtype=np.float32)

        self.reward = msg['reward']
        info = {}

        return self.observation, msg['reward'], bool(msg['done']), False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # print("reset called")

        # self.observation: ObservationDict = {"ball_velocity": np.array([0, 0, 0], dtype=np.float32),
        #                                      "ball_position": np.array([0, 0, 0], dtype=np.float32),
        #                                      "board_position": np.array([0, 0, 0], dtype=np.float32)}

        self.observation = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        self.reward = 0

        # Implement reset method
        info = {}
        return self.observation, info


if __name__ == "__main__":

    ws = websocket.WebSocket()
    ws.connect("ws://127.0.0.1:5174", timeout=5)

    env = BounceEnv(ws_connection=ws)

    check_env(env)

    ws.close()
