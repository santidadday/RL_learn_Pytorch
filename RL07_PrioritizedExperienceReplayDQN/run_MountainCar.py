import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import numpy as np

# 设置环境为MountainCar-v0, 并设置运动过程使用可视化
env = gym.make("MountainCar-v0", render_mode="human")
# 取消OpenAI Gym包装环境. 这样做是为了获取Pendulum环境的原始版本, 以便进行更灵活的设置和访问
env = env.unwrapped
# 设置存储记忆尺寸
MEMORY_SIZE = 10000

# 创建两个DQN的实例, 一个为普通DQN, 另一个为prioritized的DQN
RL_natural = DQNPrioritizedReplay(n_actions=3, n_features=2, memory_size=MEMORY_SIZE, e_greedy_increment=0.00005, prioritized=False)
RL_prio = DQNPrioritizedReplay(n_actions=3, n_features=2, memory_size=MEMORY_SIZE, e_greedy_increment=0.00005, prioritized=True)


def train(RL):
    # 创建一个初始值为0的变量, 用于追踪整个训练过程中智能体执行的总步数
    total_steps = 0

    # 创建一个空列表, 用于存储每个实验episode完成时的总步数
    steps = []

    # 创建一个空列表, 用于存储每个实验的编号, 在每次实验结束时, 将当前编号添加到列表中[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    episodes = []

    # 进行20次实验, 每次实验结束表示正难题已经成功到达目标或者无法在取得进展
    for i_episode in range(20):
        # 每次实验开始时, 通过env.reset(seed=1)重置环境, 使得实验结果可重复
        observation, _ = env.reset(seed=1)

        # 内存循环表示每一步的决策和环境交互, 直到本次实验结束
        while True:
            # 智能体根据当前状态值选择动作
            action = RL.choose_action(observation)

            # 执行选择的动作, 获取下一步的状态值、奖励、结束标志位、截断标志位
            observation_, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                # 本环境过程达到末端后获得固定奖励值10
                reward = 10

            # 将本次环境, 动作, 奖励, 下一次环境存储到经验回放缓冲区中
            RL.store_transition(observation, action, reward, observation_)

            # 总步数大于规定记忆步数进行训练
            if total_steps > MEMORY_SIZE:
                RL.learn()

            # 到达终点
            if terminated:
                print("episode ", i_episode, " finished")
                # 累计步数
                steps.append(total_steps)
                # 第i次到达终点
                episodes.append(i_episode)
                break

            # 更新位置
            observation = observation_
            total_steps += 1

        # 到达一次终点进行一次打印
        print("steps for {}th episode: {}".format(i_episode, total_steps))
    # 将episodes和steps两个一维数组合并为一个二维数组, 并返回
    return np.vstack((episodes, steps))


his_natural = train(RL_natural)
his_prio = train(RL_prio)

plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[0, :], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[0, :], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
# Y轴位总训练时间
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()
