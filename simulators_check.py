import gymnasium as gym

# checking all the envs in gym sim
mujoco_env_list = (
    "Walker2d-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "HumanoidStandup-v4",
    "Humanoid-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    # "Swimmer-v4", # it has a internal problem. you shouldn't waste your time checking it now
)

box2d_env_list = (
    "BipedalWalker-v3",
    "CarRacing-v2",
    "LunarLander-v2",
)
atari_env_list = (
    "ALE/Adventure-v5",
    "ALE/AirRaid-v5",
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
)

toy_env_list = ("CliffWalking-v0", "Blackjack-v1", "FrozenLake-v1", "Taxi-v3")


def simple_simulate(env: gym.Env):
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


# TODO swimmer didn't work
for environment_name in mujoco_env_list:
    print(environment_name)
    env = gym.make(environment_name, render_mode="human")
    simple_simulate(env)


# for environment_name in box2d_env_list:
#     print(environment_name)
#     env = gym.make(environment_name, render_mode="human")
#     simple_simulate(env)

# TODO atari didn't work
# for environment_name in atari_env_list:
#     print(environment_name)
#     env = gym.make(environment_name, render_mode="human")
#     simple_simulate(env)
# for environment_name in toy_env_list:
#     print(environment_name)
#     env = gym.make(environment_name, render_mode="human")
#     simple_simulate(env)
