from stable_baselines3 import PPO, DQN
import os
from pathlib import Path

from lib.Callbacks import TensorboardCallback


def train_agent(env, algorithm=PPO):

    env_name = env.__class__.__name__
    algorithm_name = algorithm.__name__

    models_dir = os.path.join(os.path.dirname(
        __file__), 'models', env_name + '-' + algorithm_name)
    logdir = os.path.join(os.path.dirname(
        __file__), 'logs', env_name + '-' + algorithm_name)

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
        model = algorithm('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 100000
    iters = 0

    tensorboard_callback = TensorboardCallback()

    while True:
        iters += 1

        # with ProgressBarManager(TIMESTEPS) as progress_callback:
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False, tb_log_name=f"{last_iter+TIMESTEPS * iters}",
                    callback=[tensorboard_callback])

        model.save(f"{models_dir}/{last_iter+TIMESTEPS * iters}")

        if iters > 4:
            break


if __name__ == "__main__":

    from envs.PunchEnv import PunchEnv

    env = PunchEnv()

    train_agent(env, algorithm=DQN)
