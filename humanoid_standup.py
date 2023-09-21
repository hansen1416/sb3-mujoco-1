

from stable_baselines3 import PPO

from train import train_agent
from envs.HumanoidStandupEnv import HumanoidStandupEnv


def test():

    model = PPO.load(
        "models/CustomHumanoidStandupEnv-PPO/1500000.zip")

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

    test()
