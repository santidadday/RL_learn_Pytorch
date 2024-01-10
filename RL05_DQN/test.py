from maze_env import Maze
from RL_brain import DeepQNetwork


# 该函数是执行迷宫训练的主要逻辑
def run_maze():
    step = 0
    for episode in range(300):
        # initial observation 在每个episode开始时初始化观察值, 本项目中的观测值observation和observation_为当前智能体位置
        observation = env.reset()

        while True:
            # fresh env 刷新环境, 以便可视化智能体的行为
            env.render()

            # RL choose action based on observation 使用深度Q网络基于当前观察值选择动作
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward 执行选定的动作, 并获取下一个观察值和奖励
            observation_, reward, done = env.step(action)

            # 将转换(当前观察值、动作、奖励、下一个观察值)存储到经验回放缓存中
            RL.store_transition(observation, action, reward, observation_)

            # 如果步数大于200并且是5的倍数, 执行深度Q网络的学习过程
            if (step > 200) and (step % 5 == 0):
                # cumulative experience
                RL.learn()

            # swap observation 更新当前观察值
            observation = observation_

            # break while loop when end of this episode 如果done为True, 表示智能体跳入陷阱或跳入奖励点, 当前episode结束, 跳出循环
            if done:
                break

            # 增加步数计数器
            step += 1

    # end of game 输出游戏结束信息, 销毁迷宫环境
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      )

    # 在100毫秒后执行run_maze函数
    env.after(100, run_maze)

    # 启动Tkinter主循环
    env.mainloop()

    # 可视化训练过程中的损失
    RL.plot_cost()
