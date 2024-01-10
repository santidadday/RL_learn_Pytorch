import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置NumPy可PyTorch两个库的随机数生成种子
np.random.seed(1)
torch.manual_seed(1)


# 这是一个实现优先级数(Priority Tree)的类SumTree, 通常用于优先级回放经验的实现
class SumTree(object):
    # 初始化data_pointer用于跟踪在优先级树中存储经验数据的位置, data_pointer表示当前可以存储经验数据的位置索引
    data_pointer = 0

    # 类的构造函数, 用于初始化优先级树, capacity是树的容量, 即可以存储的优先级值的最大数量
    def __init__(self, capacity):
        self.capacity = capacity  # 优先级树的容量, 即可以存储优先级值的最大数量
        self.tree = np.zeros(2 * capacity - 1)  # 创建一个2 * capacity - 1的NumPy数组用于存储优先级树的节点值, 这个数组长度是树的节点数, 其中包括叶子节点和非叶子节点
        self.data = np.zeros(capacity, dtype=object)  # 创建一个长度为capacity的NumPy数组, 用于存储优先级树中每个节点相关联的数据(即经验数据)

    # 用于向优先级树中添加新的经验数据
    def add(self, p, data):
        # 计算要插入的新的经验数据在优先级树中对应索引, self.data_pointer表示当前可用的存储位置, self.capacity - 1是因为树的节点索引是从0开始的
        tree_idx = self.data_pointer + self.capacity - 1

        # 将新的经验数据data存储到self.data数组中的当前可用位置, 这个数组用于存储所有的经验数据
        self.data[self.data_pointer] = data

        # 调用update方法, 更新树中相应的索引的节点值, p是新的经验数据的优先级值
        self.update(tree_idx, p)

        # 将data_pointer指向下一个可用的存储位置, 这是为了确保下一条经验数据存储在数组中的下一个位置
        self.data_pointer += 1

        # 检查self.data_pointer是否超过数组的容量, 如果超过表示经验数组已经装满经验数据, 需要重新利用数组空间
        if self.data_pointer >= self.capacity:
            # 回到数组的起始位置, 实现数组空间的循环利用
            self.data_pointer = 0

    # 用于更新优先级树中的节点值, 以及相应地调整树的结构
    def update(self, tree_idx, p):
        # 计算要更新的节点的优先级的变化量, 这是新的优先级值p与当前节点原有的优先级值self.tree[tree_idx]之差
        change = p - self.tree[tree_idx]

        # 将节点优先级值更新为新的优先级值p
        self.tree[tree_idx] = p

        # 进入一个循环, 不断地向树的根节点方向递归更新父节点
        while tree_idx != 0:
            # 更新当前节点的索引为其父节点的索引, 在二叉树中, 左子节点的索引为parent_index * 2 + 1, 右子节点的索引为parent_index * 2 + 2
            tree_idx = (tree_idx - 1) // 2
            # 将父节点的优先级增加变化量change, 这是为了确保整个树的结构得到正确更新, 使得从叶子节点到根节点的路径上所有节点的值都能反应新添加数据的变化
            self.tree[tree_idx] += change

    # 用于从优先级树中获取一个叶子节点, 该叶子节点的优先级大于等于给定的值v
    def get_leaf(self, v):
        # 初始化当前节点的索引为树的根节点索引
        parent_idx = 0

        # 循环找到符号条件的叶子结点
        while True:
            # 计算当前节点的左子节点索引
            cl_idx = 2 * parent_idx + 1  # left kid of the parent node
            # 计算当前节点的右子节点索引
            cr_idx = cl_idx + 1

            # 检查左子节点索引是否超出了树的范围, 如果是说明当前节点就是叶子结点, 已经没有子节点了
            if cl_idx >= len(self.tree):  # kid node is out of the tree, so parent is the leaf node
                # 将当前节点的索引作为找到叶子节点的索引
                leaf_idx = parent_idx
                break
            # 如果当前节点不是叶子结点, 表示还有子节点, 需要继续向下搜索
            else:  # downward search, always search for a higher priority node
                # 检查给定值v是否小于或等于左子节点的优先级, 如果是表示应该继续向左子节点搜索, 因为左子节点优先级更高
                if v <= self.tree[cl_idx]:
                    # 更新当前节点的索引为左子节点继续向下搜索
                    parent_idx = cl_idx
                # 如果v大于左子节点的优先级, 表示应该向右子节点搜索
                else:
                    # 减去左子节点优先级值, 表示在右子节点中搜索一个更高优先级的节点
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # 计算找到叶子结点对应的数据索引, 由于数据存储在数组self.data中, 因此需要减去self.capacity并加1
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    # 属性方法, 用于获取整个优先级树的总优先级
    @property  # 装饰器, 将total_p方法转化为一个只读属性, 当访问total_p方法时会自动调用这个方法
    def total_p(self):
        # 返回整个优先级树的总优先级, 即数组首元素
        return self.tree[0]


# 表示经验回放缓冲区
class Memory(object):  # stored as (s, a, r, s_) in SumTree
    epsilon = 0.01  # 用于计算优先级时避免零优先级, 当前计算的优先级值小于epsilon时, 使用epsilon
    alpha = 0.6  # 在计算优先级时用于将TD(时序差分)误差的重要性转化为优先级, 取值[0, 1]
    beta = 0.4  # 用于采样时进行重要性采样, 它的初始值为0.4, 在采样时逐渐增加到1.0
    beta_increment_per_sampling = 0.001  # 表示每次采用时bata值的增长步长
    abs_err_upper = 1.  # 用于剪切TD误差的绝对值, 当TD误差的绝对值超过这个上限时将其剪切为abs_err_upper

    # 构造方法的定义, 接收一个参数capacity, 表示经验回放缓冲区的容量大小
    def __init__(self, capacity):
        # 创建一个SumTree类对象, 即优先级树, 使用给定的容量capacity进行初始化
        self.tree = SumTree(capacity)

    # 用于向经验回放缓冲区添加新的经验数据
    def store(self, transition):
        # 计算经验回放缓冲区中最大的优先级值, 通过检索优先级树中最后capacity个节点优先级值, 找到其中的最大值
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        # 检查最大优先级值是否为零, 如果为零表示缓冲区中尚未存储任何数据, 此时将max_p设置为abs_err_upper即一个预先设定的上限值
        if max_p == 0:
            max_p = self.abs_err_upper
        # 调用优先级树add方法, 将新的经验数据transition存储到优先级树中. 同时将该数据的优先级设置为max_p, 确保它被赋予缓冲区最大的优先级
        self.tree.add(max_p, transition)  # set the max of p for new p

    # 用于从经验回放缓冲区中进行优先级采样
    def sample(self, n):
        # 创建三个空数组, b_idx用于存储采样的索引, b_memory用于存储对应的经验数据, ISWeights用于存储重要性采样权重
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))

        # 计算优先级段的大小, 用于请确定每个采样区间的范围
        pri_seg = self.tree.total_p / n

        # 更新重要性采样的参数beta, 确保其不超过上限值1.0
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max=1

        # 计算最小概率, 用于后续计算重要性采样权重
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculation ISweight

        # 循环采样n个样本
        for i in range(n):
            # 计算当前采样区间的起始和结束位置
            a, b = pri_seg * i, pri_seg * (i + 1)

            # 生成在当前区间内均匀分布的随机值v, 用于选择采样
            v = np.random.uniform(a, b)

            # 调用优先级树get_leaf方法, 根据随机值v获取对应的叶子节点的索引、优先级值p以及与之关联的经验数据
            idx, p, data = self.tree.get_leaf(v)

            # 计算当前经验数据的采样概率
            prob = p / self.tree.total_p

            # 计算并存储当前经验数据的重要性采样权重
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)

            # 存储当前样本的索引和对应的经验数据
            b_idx[i], b_memory[i, :] = idx, data
        # 返回采样得到的索引、经验数据和重要性采样权重
        return b_idx, b_memory, ISWeights

    # 用于批量更新优先级树中的节点优先级, tree_idx表示要更新的节点索引的数组, abs_errors表示对应节点的绝对误差的数组
    def batch_update(self, tree_idx, abs_errors):
        # 将所有绝对误差值加上一个销量, 避免零值, 这个操作是为了防止出现零误差, 因为零误差会导致重要性采样权重无法计算
        abs_errors += self.epsilon  # convert to abs and avoid 0

        # 对绝对误差进行剪切, 确保其不超过预先设定的上限
        clipped_errors = np.minimum(abs_errors.data, self.abs_err_upper)

        # 计算剪切后的绝对误差的幂, 其中alpha是一个超参数, 用于将绝对误差的重要性转化为优先级
        ps = np.power(clipped_errors, self.alpha)

        # 遍历给定的节点索引和对应的计算得到优先级值
        for ti, p in zip(tree_idx, ps):
            # 调用优先级树update方法, 根据给定的节点索引ti和对应的优先级值p来更新优先级树中的节点优先级
            self.tree.update(ti, p)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.el = nn.Linear(n_feature, n_hidden)
        self.q = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.el(x)
        x = F.relu(x)
        x = self.q(x)
        return x


class DQNPrioritizedReplay:
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9, replace_target_iter=500,
                 memory_size=10000, batch_size=32, e_greedy_increment=None, output_graph=False, prioritized=True):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized
        self.learn_step_counter = 0
        self._build_net()

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.cost_his = []

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, s_):
        if self.prioritized:  # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)  # have high priority for newly arrived transition
        else:  # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])
        if np.random.uniform() < self.epsilon:
            actions_value = self.q_eval(observation)
            action = int(torch.max(actions_value, dim=1)[1])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        # print("target params replaced\n")

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]

        if self.prioritized:
            self.abs_errors = torch.sum(torch.abs(q_target - q_eval), dim=1)
            # print("ISWeights shape: ", ISWeights.shape, 'q shape: ', ((q_target-q_eval)**2), 'q: ', (q_target-q_eval))
            loss = torch.mean(torch.mean(torch.Tensor(ISWeights) * (q_target - q_eval) ** 2, dim=1))
            self.memory.batch_update(tree_idx, self.abs_errors)
        else:
            self.loss_func = nn.MSELoss()
            loss = self.loss_func(q_eval, q_target)

        # print("loss: ", loss, self.prioritized)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.cost_his.append(loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
