# import gymnasium as gym 
# import numpy as np
# import loco_mujoco
# from loco_mujoco import LocoEnv


# mdp = LocoEnv.make("UnitreeA1.simple.perfect")

# dataset = mdp.create_dataset()
# # print(dataset['actions'])

# import gymnasium as gym
# import numpy as np

# env = gym.make('FetchSlideDense-v2', render_mode="rgb_array")
# env = gym.wrappers.RecordVideo(env, f"videos/slide", episode_trigger = lambda x: x % 2 == 0)

# n_episodes = 10
# for _ in range(n_episodes):
#     state, done = env.reset(), False
#     # env.render()
#     episode_reward = 0.0
#     for i in range(100):
#         # action = actor.act(state, device)
#         action = env.action_space.sample()
#         state, reward, done, _, _ = env.step(action)
#         # env.render()
#         episode_reward += reward
#     # episode_rewards.append(episode_reward)
# env.close()

import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym


# create the environment and task together with the reward function
env = gym.make("LocoMujoco", env_name="UnitreeA1.simple.perfect", render_mode = "rgb_array")
env = gym.wrappers.RecordVideo(env, f"videos/unitreeA1")#, episode_trigger = lambda x: x % 10 == 0)
action_dim = env.action_space.shape[0]

env.reset()
env.render()
terminated = False
i = 0

while True:
    if i == 1000 or terminated:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, terminated, truncated, info = env.step(action)

    # HERE is your favorite RL algorithm

    env.render()
    i += 1