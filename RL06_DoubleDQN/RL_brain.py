import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置NumPy随机数种子为1, 以确保随机数的可重复性
np.random.seed(1)
# 设置PyTorch的随机数种子wield1, 同样是为了确保随机数的可重复性
torch.manual_seed(1)


# 定义一个Net神经网络类, 一个输入层, 一个隐藏层, 一个输出层
class Net(nn.Module):
    # 定义了初始化函数, 用于设置神经网络的结构. 在这里n_feature表示输入特征的数量, n_hidden表示隐藏层的节点数, n_output表示输出层的节点数
    def __init__(self, n_feature, n_hidden, n_output):
        # 调用父类的初始化函数, 确保正确初始化神经网络
        super(Net, self).__init__()
        # 创建了一个线性层(全连接层), 将输入特征的数量映射到隐藏层的节点数
        self.el = nn.Linear(n_feature, n_hidden)
        # 创建了一个线性层, 将隐藏层的节点数映射到输出层的节点数
        self.q = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 线性变化
        x = self.el(x)
        # 激活函数
        x = F.relu(x)
        # 将ReLU激活后的输出经过输出层的线性变换, 得到最终的输出
        x = self.q(x)
        return x


# DoubleDQN引入了对目标Q值的两次估计, 以减轻DQN中对最大动作值的过度估计(overestimation)的问题
# 在DoubleDQN中, 使用行为网路选择当前状态下最优的动作, 然后使用目标网络评估这个动作的Q值, 这种方式有助于减少过度估计的影响
class DoubleDQN():
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200, memory_size=3000, batch_size=32, e_greedy_increment=None, double_q=True):
        self.n_actions = n_actions  # 动作数量(输入层节点数)
        self.n_features = n_features  # 状态数量(输出层节点数)
        self.n_hidden = n_hidden  # 隐藏层的节点数
        self.lr = learning_rate  # 神经网络学习速度
        self.gamma = reward_decay  # 强化学习中Q-learning折扣因子
        self.epsilon_max = e_greedy  # 探索策略中的最大探索概率
        self.replace_target_iter = replace_target_iter  # 定义了多少步更新一次目标网络
        self.memory_size = memory_size  # 经验回放存储的记忆容量
        self.batch_size = batch_size  # 定义了每次从记忆中采样的样本数量
        self.epsilon_increment = e_greedy_increment  # 探索概率的增量, 如果为None则初始探索概率为e_greedy_max, 否则会逐步增加
        self.double_q = double_q  # 用于判断本次训练采用的是DoubleDQN还是DQN, 提高代码复用性

        if e_greedy_increment is not None:
            # 判断探索策略概率是否随之训练轮次的推移变化
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_max

        self.learn_step_counter = 0  # 本轮学习总步数
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # 创建一个大小为memory_size, n_features * 2 + 2的二维数组, n_features * 2 + 2表示, 当前状态特征数量, 下一状态特征数量, 奖励值, 动作
        self._build_net()  # 构建网络
        self.cost_his = []  # 累计损失

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)  # 该网络用于评估当前状态的Q值
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)  # 该网络用于评估目标状态的Q值. 这个网络的参数将在一定训练的步数后, 复制q_eval的参数进行更新
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)  # 设置优化器为RMSprop
        self.loss_func = nn.MSELoss()  # 设置损失函数为均方差

    # 定义存储经验的方法
    def store_transition(self, s, a, r, s_):
        # 如果memory_counter属性不存在, 就初始化为0, 类似与单例
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 将当前状态、动作、奖励和下一个状态合并为一个数组, 当前状态和下一个状态在本模型中为数组, 动作和奖励为一个float类型数据
        transition = np.hstack((s, [a, r], s_))

        # 计算要替换的记忆索引, 采用循环缓冲区的方式, 确保记忆不断循环利用
        index = self.memory_counter % self.memory_size
        # 将新的经验数据替换到记忆中的对应位置
        self.memory[index, :] = transition
        # 更新记忆计数器, 表示存储了一步新的经验
        self.memory_counter += 1

    # 定义了选择动作的方法, 接受当前状态作为输入
    def choose_action(self, observation):
        # 如果 observation 的第一个元素是数组，则转换为 NumPy 数组. 确保输入的数据类型为NumPy数组
        if isinstance(observation[0], np.ndarray):
            # 不会经历这个, 在reset阶段已经设置为
            observation = np.array(observation[0])

        # 将NumPy类型的数组转换为PyTorch的Tensor格式, 并在第一个维度上增加一个维度. 这是为了适应神经网络输入的格式
        observation = torch.Tensor(observation[np.newaxis, :])

        # 使用q_eval网络输入observation并输出对应的动作值(Q值)
        actions_value = self.q_eval(observation)

        # 从给定张量action_value中选择最大的索引, 这段代码返回的是actions_value中每一行的最大值所在的列的索引
        action = torch.max(actions_value, dim=1)[1]  # record action value it get

        if not hasattr(self, 'q'):
            # 判断是否该类对象中是否含有q列表, 如果没有则进行创建, 并创建一个running_q值
            self.q = []
            self.running_q = 0

        # 强化学习计算q值, 采用滑动平均的方式更新running_q. 这个值被用于记录Q值的变化趋势
        self.running_q = self.running_q * 0.99 + 0.01 * torch.max(actions_value, dim=1)[0]
        # 将running_q添加到q表
        self.q.append(self.running_q)

        # 根据探索概率self.epsilon随机选择动作, 如果随机数大于探索概率, 则随机选择一个动作. 如果选择Q值最大的动作
        if np.random.uniform() > self.epsilon:  # randomly choose action
            action = np.random.randint(0, self.n_actions)

        # 返回的action可能是[0, 10]中的任意整数
        return action

    # 用于执行深度强化学习的训练过程
    def learn(self):
        # 如果学习步数计数器能够整除replace_target_iter则执行以下替换网络操作. 目标网络的参数在一定步数间隔内更新, 以提高稳定性
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 通过加载q_eval网络的参数来更新目标网络q_target的参数
            self.q_target.load_state_dict(self.q_eval.state_dict())
            # 打印提示信息, 表示目标网络参数已被替换
            print("target params replaced")

        # 如果记忆计数器大约记忆容量, 则从memory_size中随机选择大小为batch_size的样本
        if self.memory_counter > self.memory_size:
            # 在[0, self.memory_size)范围中选择self.batch_size个样本, 返回的是一个大小为self.batch_size的一维数组
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # 如果记忆计数器小于等于记忆容量, 则从memory_counter中随机选择大小为batch_size的样本
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # 从记忆中提取随机选择的样本, 从memory数组中选取sample_index数组中指定的几行 所有数据作为batch_memory数组
        batch_memory = self.memory[sample_index, :]

        # 分别对目标网络q_target和当前网络q_eval计算下一状态s_(t+1)对应的Q值
        # 其传入的Tensor数据为batch_memory中的所有行中倒数第self.n_features到最后一列的二维数组转为的PyTorch张量数据
        # 其传出为PyTorch张量数据(self.batch_size, n_actions)
        q_next, q_eval4next = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(torch.Tensor(batch_memory[:, -self.n_features:]))

        # 当前网络q_eval计算当前装填s_(t)对应的Q值
        q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))

        # 复制当前状态s_(t)对应的Q值, 用于计算目标Q值
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        # 创建一个数组, 用于表示批处理中的索引
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # 提取样本中的动作索引, 作为当前状态s_(t)执行的动作
        # 返回值为一个整数数组, 其中包括了每个记忆中的动作的离散索引
        eval_act_index = batch_memory[:, self.n_features].astype(int)

        # 提取样本中的奖励值
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])

        # 如果使用DoubleDQN
        if self.double_q:
            # 从q_act4next中选择最大的索引, 表示在下一状态s_(t+1)中选择的动作
            max_act4next = torch.max(q_eval4next, dim=1)[1]
            # 从目标网络的Q值中提取在s_(t+1)中选择的动作对应的Q值
            selected_q_next = q_next[batch_index, max_act4next]
        # 如果使用DQN
        else:
            # 直接从目标网络的Q值中选择最大的Q值
            selected_q_next = torch.max(q_next, dim=1)[0]

        # 根据Q-learning更新目标Q值
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # 计算Q值的损失, 使用均方差损失函数
        loss = self.loss_func(q_eval, q_target)

        # 将优化器的梯度清零
        self.optimizer.zero_grad()

        # 反向传播计算梯度
        loss.backward()

        # 根据梯度更新网络参数
        self.optimizer.step()

        # 将损失值添加到损失历史列表中, 用于后续分析
        self.cost_his.append(loss)

        # 更新探索概率epsilon. 如果启用了探索概率的递增并且当前概率小于最大概率, 则递增概率. 否则保持不变
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # 更新学习步数计数器
        self.learn_step_counter += 1
