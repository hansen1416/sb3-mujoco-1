
import os
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from envs.HumanoidStandupEnv import HumanoidStandupEnv
from utils.functions import train_agent, linear_schedule
from lib.Callbacks import TensorboardCallback
from stable_baselines3.common.evaluation import evaluate_policy


def test(model_path="models/CustomHumanoidStandupEnv-PPO/1500000.zip"):

    model = PPO.load(model_path)

    # print(model)
    env = HumanoidStandupEnv(render_mode="human")
    obs, _ = env.reset()

    # print(obs)
    while True:
        action, _ = model.predict(obs)

        obs, rewards, dones, truncate, info = env.step(action)

        env.render()

        print("action: {}, reward: {}".format(action, rewards))


if __name__ == "__main__":

    """
    learning_rate: Union[float, Schedule] = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: Union[float, Schedule] = 0.2,
    clip_range_vf: Union[None, float, Schedule] = None,
    normalize_advantage: bool = True,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    target_kl: Optional[float] = None,
    stats_window_size: int = 100,
    tensorboard_log: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[th.device, str] = "auto",
    _init_setup_model: bool = True,
    """

    algorithm = PPO
    env = HumanoidStandupEnv()
    params = {
        "learning_rate": linear_schedule(0.01),
    }

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

    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env])
    env.reset()

    if len(paths) > 0:
        # get last model file
        last_model = paths[-1]

        # get last iteration
        last_iter = int(os.path.splitext(last_model.name)[0])

        last_model = algorithm.load(last_model, env, verbose=1,
                                    tensorboard_log=logdir, **params)

    if last_model:
        model = last_model
    else:
        # model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)
        model = algorithm('MlpPolicy', env, verbose=1,
                          tensorboard_log=logdir, **params)

    TIMESTEPS = 1000

    # tensorboard_callback = TensorboardCallback()
    # Create the callback object
    # eval_callback = EvalCallback(eval_env=env,
    #                              best_model_save_path=models_dir,
    #                              log_path=logdir,
    #                              eval_freq=100,
    #                              #  deterministic=True,
    #                              render=False
    #                              )


    # with ProgressBarManager(TIMESTEPS) as progress_callback:
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=f"{last_iter+TIMESTEPS}",
                # callback=[tensorboard_callback, eval_callback]
                # callback=[eval_callback]
                )
    
    policy = model.policy
    # Retrieve the environment
    env = model.get_env()
    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # reward_list_base, episode_list_base = evaluate_policy(BasePolicy, model.get_env(), n_eval_episodes=10, return_episode_rewards=True)

    # model.save(f"{models_dir}/{last_iter + TIMESTEPS}.zip")

    # print(eval_callback.best_mean_reward)
    # print(eval_callback.last_mean_reward)

    # test()
