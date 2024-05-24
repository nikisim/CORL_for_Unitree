import gym
import gym_UR5_FetchPush


env = gym.make('gym_UR5_FetchPush/UR5_FetchPushEnv-v0', render=False)



for episode in range(15):
    obs ,_= env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, rew, done, _, info = env.step(action)
        print('*****')
        # # print("obs_shape:", obs['observation'].shape())
        print("obs:", obs)
        print('*****')
        # print("rew:", rew)
        # print('*****')
        # print("action:", action)
        # print('*****')
        if done:
            break
env.close()