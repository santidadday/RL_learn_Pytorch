import gym
import warnings
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
# 创建Pendulum环境, 该环境为一个倒立摆任务, 使用物理引擎进行模拟, g=9.81设置重力加速度为标准地球重力, render_mode="human"设置渲染模式为人类可视化
env = gym.make("Pendulum-v1", g=9.81, render_mode="human")
# 取消OpenAI Gym包装环境. 这样做是为了获取Pendulum环境的原始版本, 以便进行更灵活的设置和访问
env = env.unwrapped
# 设置经验回访记忆库的大小
MEMORY_SIZE = 3000
# 离散空间动作, 本项目将动作分为11个-2~2之间的扭矩施加在摆锤上
ACTION_SPACE = 11

# 创建一个双Q学习的代理naturel_DQN, 指定了动作空间大小为ACTION_SPACE, 状态特征数3, 记忆库大小3000, 探索概率递增值0.001, 是否使用DoubleDQN标志
naturel_DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE, e_greedy_increment=0.001, double_q=False)
double_DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE, e_greedy_increment=0.001, double_q=True)


# 定义一个训练函数, 接受一个双Q学习代理RL作为输入
def train(RL):
    # 初始化总步数计数器
    total_steps = 0
    # 通过调用环境的reset方法初始化环境, 并获取初始观测状态observation
    observation, _ = env.reset(seed=1)

    # 进入训练循环
    while True:
        # 如果已经训练了一段时间, 超过经验回放记忆库的容量, 就开始渲染环境以观察智能体行为
        if total_steps - MEMORY_SIZE > 8000:
            # 显示环境
            env.render()  # show the game when trained for some time

        # 通过调用代理的choose_action方法选择一个动作
        action = RL.choose_action(observation)
        print(action)

        # 将离散的动作值转换为连续范围在[-2, 2]的浮点数
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions

        # 执行动作, 获取下一个观测值状态, 奖励值, 是否结束标志
        observation_, reward, done, _, _ = env.step(np.array([f_action]))

        # 将奖励值归一化到范围(-1, 0)其中在摆正状态的奖励为0
        reward /= 10  # normalize to a range of (-1,0). r = 0 when get upright

        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.
        # 将当前状态、动作、奖励和下一个状态的转移存储到经验回放记忆库中
        RL.store_transition(observation, action, reward, observation_)

        # 当总步数超过经验回访记忆库容量后, 调用代理的learn方法学习
        if total_steps > MEMORY_SIZE:
            RL.learn()

        # 如果已经进行了足够多的训练步数, 结束训练循环
        if total_steps - MEMORY_SIZE > 20000:
            break

        # 更新当前状态为下一状态
        observation = observation_
        # 增加总步数计数器
        total_steps += 1

    return RL.q


# 使用train函数训练普通DQN代理, 返回代理的Q值。
q_natural = train(naturel_DQN)
# 将Q值张量转换为Python列表, 以便后续绘图
q_natural_array = [tensor.item() for tensor in q_natural]
# 使用train函数训练DoubleDQN代理, 返回代理的Q值
q_double = train(double_DQN)
# 将Q值张量转换为Python列表, 以便后续绘图
q_double_array = [tensor.item() for tensor in q_double]

# 绘制普通DQN的曲线使用红色标注
plt.plot(q_natural_array, c='r', label='natural')
# 绘制DoubleDQN的曲线使用蓝线标注
plt.plot(q_double_array, c='b', label='double')
# 在图中显示图例，位置选择最佳位置
plt.legend(loc='best')
# Y轴标签
plt.ylabel('Q eval')
# X轴标签
plt.xlabel('training steps')
# 添加网格
plt.grid()
# plt展示
plt.show()
# 环境关闭
env.close()
