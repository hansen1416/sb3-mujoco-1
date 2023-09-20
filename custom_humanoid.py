import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv


class CustomHumanoidEnv(HumanoidEnv):
    def __init__(self, render_mode='human', **kwargs):
        super().__init__(**kwargs)
        self.render_mode = render_mode

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            self.render_mode = mode
        return super().render(**kwargs)


def train():
    env = gym.make('Humanoid-v4')

    env = DummyVecEnv([lambda: env])

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000000)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


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


if __name__ == "__main__":
    demo()
