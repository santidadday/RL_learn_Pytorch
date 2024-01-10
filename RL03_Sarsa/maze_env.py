import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 80  # pixels
UNIT_HALF = 40
HELL_RADIUS = 30
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT_HALF, UNIT_HALF])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(hell1_center[0] - HELL_RADIUS, hell1_center[1] - HELL_RADIUS, hell1_center[0] + HELL_RADIUS, hell1_center[1] + HELL_RADIUS, fill='black')

        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(hell2_center[0] - HELL_RADIUS, hell2_center[1] - HELL_RADIUS, hell2_center[0] + HELL_RADIUS, hell2_center[1] + HELL_RADIUS, fill='black')

        # # hell
        # hell3_center = origin + np.array([UNIT * 2, UNIT * 6])
        # self.hell3 = self.canvas.create_rectangle(hell3_center[0] - HELL_RADIUS, hell3_center[1] - HELL_RADIUS, hell3_center[0] + HELL_RADIUS, hell3_center[1] + HELL_RADIUS, fill='black')
        #
        # # hell
        # hell4_center = origin + np.array([UNIT * 6, UNIT * 2])
        # self.hell4 = self.canvas.create_rectangle(hell4_center[0] - HELL_RADIUS, hell4_center[1] - HELL_RADIUS, hell4_center[0] + HELL_RADIUS, hell4_center[1] + HELL_RADIUS, fill='black')
        #
        # # hell
        # hell5_center = origin + np.array([UNIT * 4, UNIT * 4])
        # self.hell5 = self.canvas.create_rectangle(hell5_center[0] - HELL_RADIUS, hell5_center[1] - HELL_RADIUS, hell5_center[0] + HELL_RADIUS, hell5_center[1] + HELL_RADIUS, fill='black')
        #
        # # hell
        # hell6_center = origin + np.array([UNIT * 4, UNIT * 1])
        # self.hell6 = self.canvas.create_rectangle(hell6_center[0] - HELL_RADIUS, hell6_center[1] - HELL_RADIUS, hell6_center[0] + HELL_RADIUS, hell6_center[1] + HELL_RADIUS, fill='black')
        #
        # # hell
        # hell7_center = origin + np.array([UNIT * 1, UNIT * 3])
        # self.hell7 = self.canvas.create_rectangle(hell7_center[0] - HELL_RADIUS, hell7_center[1] - HELL_RADIUS, hell7_center[0] + HELL_RADIUS, hell7_center[1] + HELL_RADIUS, fill='black')
        #
        # # hell
        # hell8_center = origin + np.array([UNIT * 2, UNIT * 4])
        # self.hell8 = self.canvas.create_rectangle(hell8_center[0] - HELL_RADIUS, hell8_center[1] - HELL_RADIUS, hell8_center[0] + HELL_RADIUS, hell8_center[1] + HELL_RADIUS, fill='black')
        #
        # # hell
        # hell9_center = origin + np.array([UNIT * 3, UNIT * 2])
        # self.hell9 = self.canvas.create_rectangle(hell9_center[0] - HELL_RADIUS, hell9_center[1] - HELL_RADIUS, hell9_center[0] + HELL_RADIUS, hell9_center[1] + HELL_RADIUS, fill='black')

        # create oval
        oval_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.oval = self.canvas.create_oval(oval_center[0] - HELL_RADIUS, oval_center[1] - HELL_RADIUS, oval_center[0] + HELL_RADIUS, oval_center[1] + HELL_RADIUS, fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(origin[0] - HELL_RADIUS, origin[1] - HELL_RADIUS, origin[0] + HELL_RADIUS, origin[1] + HELL_RADIUS, fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([UNIT_HALF, UNIT_HALF])
        self.rect = self.canvas.create_rectangle(origin[0] - HELL_RADIUS, origin[1] - HELL_RADIUS, origin[0] + HELL_RADIUS, origin[1] + HELL_RADIUS, fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
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

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1),
                    self.canvas.coords(self.hell2),
                    # self.canvas.coords(self.hell3),
                    # self.canvas.coords(self.hell4),
                    # self.canvas.coords(self.hell5),
                    # self.canvas.coords(self.hell6),
                    # self.canvas.coords(self.hell7),
                    # self.canvas.coords(self.hell8),
                    # self.canvas.coords(self.hell9)
                    ]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.02)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
