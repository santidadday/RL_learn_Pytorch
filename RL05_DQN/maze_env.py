import numpy as np
import time
import sys

# 根据python版本导入适当的Tkinter库
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 80  # 单元格大小
UNIT_HALF = 40  # 单元格一半大小
HELL_RADIUS = 30  # 危险区域的半径
MAZE_H = 4  # 迷宫高度(格子数)
MAZE_W = 4  # 迷宫宽度(格子数)


class Maze(tk.Tk, object):
    # 迷宫类定义
    def __init__(self):
        super(Maze, self).__init__()  # 继承父类tk.Tk, 用于初始化Tkinter窗口
        self.action_space = ['u', 'd', 'l', 'r']  # 创建一个包含上下左右四个动作的列表, 表示代理可以执行的动作
        self.n_actions = len(self.action_space)  # 计算动作空间的大小, 即动作数量
        self.n_features = 2  # 设置观测空间的特征数量, 这里为2. 在这个迷宫中, 观测值是代理位置相对于目标位置的相对坐标
        self.title('maze')  # 设置Tkinter窗口的标题为"maze"
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))  # 设置Tkinter窗口大小, 根据迷宫的高度('MAZE_H')和宽度('MAZE_W')以及单元格大小('UNIT')计算
        self._build_maze()  # 调用内部'_build_maze()'方法, 构建迷宫的图形界面

    # 初始化迷宫界面
    def _build_maze(self):
        # 创建一个Tkinter画布, 用于绘制迷宫
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # create grids 在画布上绘制迷宫的网格, 包括垂直和水平的线
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin 创建起点、目标点、陷阱图形和元素, 分表用红色矩形、黄色圆形、黑色矩形来表示
        origin = np.array([UNIT_HALF, UNIT_HALF])

        # hell 设置中心点, 从中心点向外延伸为矩形的四条边
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(hell1_center[0] - HELL_RADIUS, hell1_center[1] - HELL_RADIUS, hell1_center[0] + HELL_RADIUS, hell1_center[1] + HELL_RADIUS, fill='black')

        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(hell2_center[0] - HELL_RADIUS, hell2_center[1] - HELL_RADIUS, hell2_center[0] + HELL_RADIUS, hell2_center[1] + HELL_RADIUS, fill='black')

        # create oval 圆形设置时需要调用椭圆形创建函数, 将椭圆形四条切线设置好后即为圆形
        oval_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.oval = self.canvas.create_oval(oval_center[0] - HELL_RADIUS, oval_center[1] - HELL_RADIUS, oval_center[0] + HELL_RADIUS, oval_center[1] + HELL_RADIUS, fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(origin[0] - HELL_RADIUS, origin[1] - HELL_RADIUS, origin[0] + HELL_RADIUS, origin[1] + HELL_RADIUS, fill='red')

        # pack all 将画布放置到Tkinter窗口中
        self.canvas.pack()

    # 重置迷宫状态
    def reset(self):
        # 更新所有任务, 确保在删除矩阵之前, 所有的Tkinter任务都已完成
        self.update_idletasks()

        # 引入短暂延时, 确保所有的更新和删除操作都已经完成
        time.sleep(0.1)

        # 删除之前的智能体矩形, 以便在重新设置初始状态时重新创建
        self.canvas.delete(self.rect)

        # 定义新的起始点坐标, 即矩形的初始位置
        origin = np.array([UNIT_HALF, UNIT_HALF])

        # 在新的起始点创建一个红色矩形, 表示智能体的位置
        self.rect = self.canvas.create_rectangle(origin[0] - HELL_RADIUS, origin[1] - HELL_RADIUS, origin[0] + HELL_RADIUS, origin[1] + HELL_RADIUS, fill='red')

        # 计算并返回观察值, 这个观察值表示智能体与目标点的相对位置
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)

    def step(self, action):
        # 获取当前智能体的位置
        s = self.canvas.coords(self.rect)

        # 创建一个base_action的NumPy数组, 用于存储智能体的运动方向, 其中位置是画布中智能体中心位置
        base_action = np.array([0, 0])

        # 通过输入的action更新base_action
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 移动智能体, 根据base_action更新矩阵位置
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        # 取智能体移动后的新坐标, 表示下一个状态
        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            # 智能体进入黄色获胜点, 获得正奖励
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1),
                             self.canvas.coords(self.hell2),
                             ]:
            # 智能体进入黑色失败点, 获得负奖励
            reward = -1
            done = True
        else:
            # 智能体在白色迷宫位置中进行, 不获得奖励
            reward = 0
            done = False

        # 智能体相对于目标点的位置，作为下一个状态的观察值
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)

        return s_, reward, done

    # Tkinter 中的一个方法，用于处理所有挂起的事件，这包括重绘窗口。它会触发窗口的更新，确保显示的内容是最新的
    def render(self):
        self.update()

    def destroy(self):
        if hasattr(self, 'canvas'):
            super(Maze, self).destroy()
