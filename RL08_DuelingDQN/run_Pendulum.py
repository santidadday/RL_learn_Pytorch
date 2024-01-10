import gym
from RL_brain import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Pendulum-v1", render_mode="human")
env = env.unwrapped
env.action_space.seed(1)
env.observation_space.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 25

natural_DQN = DuelingDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE, e_greedy_increment=0.001, dueling=False)
dueling_DQN = DuelingDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE, e_greedy_increment=0.001, dueling=True)


def train(RL):
    # 每个回合获得的奖励值+前一个回合已经获得奖励
    acc_r = [0]
    total_steps = 0
    observation, _ = env.reset()
    while True:
        action = RL.choose_action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
        observation_, reward, terminated, truncated, _ = env.step(np.array([f_action]))

        reward /= 10

        acc_r.append(reward + acc_r[-1])

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()
        if total_steps - MEMORY_SIZE > 15000:
            break
        observation = observation_
        total_steps += 1

    # 返回
    return RL.cost_his, acc_r


c_natural, r_natural = train(natural_DQN)
c_natural_array = [tensor.item() for tensor in c_natural]
print("start training dueling DQN! ")
c_dueling, r_dueling = train(dueling_DQN)
c_dueling_array = [tensor.item() for tensor in c_dueling]

plt.figure(1)
plt.plot(np.array(c_natural_array), c='r', label='natural')
plt.plot(np.array(c_dueling_array), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()
