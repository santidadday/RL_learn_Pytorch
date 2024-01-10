import numpy as np  # 用于处理数值计算
import torch  # 深度学习框架
import torch.nn as nn  # 提供神经网络模块
import torch.nn.functional as F  # 提供构建神经网络的函数
import matplotlib.pyplot as plt  # 用于绘图

# 设置随机种子, 确保实验的可重复性
np.random.seed(1)
torch.manual_seed(1)


# 定义神将网络模型, n_feature为输入层, n_hidden为隐藏层, n_output位输出层
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承nn.Module父类
        self.el = nn.Linear(n_feature, n_hidden)
        self.q = nn.Linear(n_hidden, n_output)

    # 前向传播方法
    def forward(self, x):
        x = self.el(x)  # 输入线性变化
        x = F.relu(x)  # 激活函数ReLU
        x = self.q(x)  # 线性变换输出
        return x


class DeepQNetwork():
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
                 ):
        self.n_actions = n_actions  # 动作的数量(输入层节点数)
        self.n_features = n_features  # 状态数量(输出层节点数)
        self.n_hidden = n_hidden  # 隐藏层的节点数
        self.lr = learning_rate  # 神经网络学习速度
        self.gamma = reward_decay  # 强化学习中Q-learning折扣因子
        self.epsilon_max = e_greedy  # 探索策略中的最大探索概率
        self.replace_target_iter = replace_target_iter  # 定义了多少步更新一次目标网络
        self.memory_size = memory_size  # 经验回放存储的记忆容量
        self.batch_size = batch_size  # 定义了每次从记忆中采样的样本数量
        self.epsilon_increment = e_greedy_increment  # 探索概率的增量, 如果为None则初始探索概率为e_greedy_max, 否则会逐步增加
        if e_greedy_increment is not None:
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_max

        # 记录学习步数的计数器
        self.learn_step_counter = 0

        # 初始化经验回放的记忆数组 [s, a, r, s_], 用于存储过去的状态、动作、奖励和下一个状态
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # 定义损失函数为均方差
        self.loss_func = nn.MSELoss()

        # 记录损失的历史列表
        self.cost_his = []

        # 构建神经网络
        self._build_net()

    def _build_net(self):
        # 创建评估网络
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        # 创建目标网络
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        # 设置优化器, 使用RMSprop优化器, 用于评估网络的参数, 学习效率为lr
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

    # 用于将经验存放到经验回放的记忆中
    def store_transition(self, s, a, r, s_):
        # 如果memory_counter属性不存在, 就初始化为0, 类似与单例
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # 将状态、动作、奖励和下一个状态合并为一个数组
        transition = np.hstack((s, [a, r], s_))
        # 将新的经验替换到记忆中
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        # 更新记忆计数器
        self.memory_counter += 1

    def choose_action(self, observation):
        # 将observation转化为PyTorch的Tensor, 增加一个维度
        observation = torch.Tensor(observation[np.newaxis, :])

        # 如果随机数小于当前探索概率epsilon, 进行探索
        if np.random.uniform() < self.epsilon:
            # 使用评估网络获取各个动作的Q值
            actions_value = self.q_eval(observation)
            # 选择Q值最大的动作
            action = np.argmax(actions_value.detach().numpy())
        else:
            # 随机选择一个动作
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 检查是否替换目标网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 用评估网络的参数更新目标网络的参数
            self.q_target.load_state_dict(self.q_eval.state_dict())

        # 从记忆中随机采样一个批次的经验
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 获取目标网路在下一个状态s_(t + 1)选择动作
        q_next = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:]))
        q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))
        # 用于计算目标Q值, 需要对q_eval进行拷贝, 因为这个操作可以保持未选择的Q值不变
        # 因此当我们计算q_target - q_eval时, 这些未选择的Q值变为零, 不影响损失的计算
        q_target = torch.Tensor(q_eval.detach().numpy().copy())

        # 获取批次中每个样本选择的动作的索引
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)

        # 获取每个样本的奖励
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])

        # 更新目标Q值
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]

        # 计算损失
        loss = self.loss_func(q_eval, q_target)

        # 梯度清零
        self.optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新神经网络的参数
        self.optimizer.step()

        # 增加探索概率epsilon
        self.cost_his.append(loss)
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon_increment + self.epsilon
        else:
            self.epsilon = self.epsilon_max

        # 增加学习步数计数器
        self.learn_step_counter += 1

    def plot_cost(self):
        # fig, axes = plt.subplots(nrows=2, ncols=2)

        # 绘制损失图
        # axes[0, 0].plot(np.arange(len(self.cost_his)), [const.detach().numpy() for const in self.cost_his])
        # axes[0, 0].set_title("loss curve")
        # axes[0, 0].set_xlabel("training steps")
        # axes[0, 0].set_ylabel("Cost")

        # 绘制损失的变化曲线
        plt.plot(np.arange(len(self.cost_his)), [const.detach().numpy() for const in self.cost_his])
        # 设置图标的标签
        plt.ylabel('Cost')
        plt.xlabel('training steps')



        # 显示图标
        plt.show()
