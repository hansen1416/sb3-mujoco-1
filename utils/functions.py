import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from lib.Callbacks import TensorboardCallback


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def point_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def train_agent(env, algorithm, params={}):

    env_name = env.__class__.__name__
    algorithm_name = algorithm.__name__

    models_dir = os.path.join(os.path.dirname(
        __file__), '..', 'models', env_name + '-' + algorithm_name)
    logdir = os.path.join(os.path.dirname(
        __file__), '..', 'logs', env_name + '-' + algorithm_name)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    paths = sorted(Path(models_dir).iterdir(), key=os.path.getmtime)

    last_model = None
    last_iter = 0

    env.reset()

    if len(paths) > 0:
        # get last model file
        last_model = paths[-1]

        # get last iteration
        last_iter = int(os.path.splitext(last_model.name)[0])

        last_model = algorithm.load(last_model, env, verbose=1,
                                    tensorboard_log=logdir)

    if last_model:
        model = last_model
    else:

        # model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)
        model = algorithm('MlpPolicy', env, verbose=1,
                          tensorboard_log=logdir, **params)

    TIMESTEPS = 100000
    # iters = 0

    tensorboard_callback = TensorboardCallback()
    # Create the callback object
    eval_callback = EvalCallback(eval_env=Monitor(env), best_model_save_path=models_dir,
                                 log_path=logdir, eval_freq=1000,
                                 deterministic=True, render=False)

    # with ProgressBarManager(TIMESTEPS) as progress_callback:
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, 
                # tb_log_name=f"{last_iter+TIMESTEPS * iters}",
                callback=[tensorboard_callback, eval_callback])

    model.save(f"{models_dir}")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":

    pass
