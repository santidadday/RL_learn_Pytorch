import time
import numpy as np
import pandas as pd

np.random.seed(2)  # 随机数

N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy police: 90%选择最优动作, 10%选择随机动作
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount factor: 未来奖励的衰减率
MAX_EPISODES = 13  # maximum episodes: 最大回合数
FRESH_TIME = 0.1  # fresh time for one move: 规定每步时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    # print(table)
    return table


# build_q_table(N_STATES, ACTIONS)
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # 判断是选择最优动作还是选择随机动作, 初始化状态全部为零时选择随机
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R  # 返回下一个执行完之后的位置和获得的奖励


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\n{}'.format(interaction))
        time.sleep(2)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)  # 将数组转化为字符串, 以字符串形式输出
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    print(q_table)
    for episode in range(MAX_EPISODES):
        # 每个回合的初识状态
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)

        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                q_target = R  # next state is terminal没有下一步
                is_terminated = True  # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter + 1)
            step_counter += 1
        print(q_table)
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print(q_table)
    print('\nQ-table:\n')
    print(q_table)
