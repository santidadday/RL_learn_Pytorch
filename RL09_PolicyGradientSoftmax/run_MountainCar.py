import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

RENDER = False
DISPLAY_REWARD_THRESHOLD = -2000

env = gym.make("MountainCar-v0", render_mode="human")
env = env.unwrapped
env.action_space.seed(1)
env.observation_space.seed(1)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995
)

for i_episode in range(1000):
    observation, _ = env.reset()

    while True:
        if RENDER:
            env.render()
        action = RL.choose_action(observation)

        observation_, reward, done, _, _ = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            # 本次训练完成, 进行学习本次训练中的内容
            ep_rs_sum = sum(RL.ep_rs)
            if "running_reward" not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            print("episode: ", i_episode, "reward: ", int(running_reward))

            vt = RL.learn()

            if i_episode == 300:
                plt.plot(vt)
                plt.xlabel("episode steps")
                plt.ylabel("normalized state-action value")
                plt.show()
            break

        observation = observation_
