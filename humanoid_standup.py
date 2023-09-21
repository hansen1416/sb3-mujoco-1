

from stable_baselines3 import PPO

from envs.HumanoidStandupEnv import HumanoidStandupEnv
from utils.functions import train_agent

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
    # demo()

    train_agent(HumanoidStandupEnv(), algorithm=PPO)

    # test()
