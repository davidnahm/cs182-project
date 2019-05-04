import random
import gym
import numpy as np
import tkinter
import itertools
import time

class GridExplore:
    grid_pixels = 50

    def _valid_squares(self, x, y):
        ns = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [n for n in ns if (0 <= n[0] < self.size_x and 0 <= n[1] < self.size_y)]

    def _neighbors(self, x, y, walls = False):
        return [n for n in self._valid_squares(x, y) if walls != self.grid[n[0]][n[1]]]
    
    def _is_open(self, x, y):
        return 0 <= x < self.size_x and 0 <= y < self.size_y and self.grid[x][y]

    def __init__(self, size_x, size_y, max_allowed_step_ratio = 2.0, include_last_action_and_reward = True):
        # Modified version of randomized Prim's algorithm from
        # https://en.wikipedia.org/wiki/Maze_generation_algorithm
        self.size_x = size_x
        self.size_y = size_y
        self.action_space = gym.spaces.Discrete(4)
        self.include_last_action_and_reward = include_last_action_and_reward

        self.obs_n = 4
        if include_last_action_and_reward:
            self.obs_n += 5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (self.obs_n,), dtype = np.float32)

        self.grid = [[False for _ in range(size_y)] for _ in range(size_x)]
        gen_x, gen_y = random.randint(0, size_x - 1), random.randint(0, size_y - 1)
        self.grid[gen_x][gen_y] = True
        candidates = set(self._valid_squares(gen_x, gen_y))
        self.open_squares = []
        while len(candidates) > 0:
            candidate = random.sample(candidates, 1)[0]
            if len(self._neighbors(*candidate)) < 2:
                candidates.update(self._neighbors(*candidate, walls = True))
                self.grid[candidate[0]][candidate[1]] = True
                self.open_squares.append(candidate)
            candidates.remove(candidate)
        
        self.max_allowed_steps = int(max_allowed_step_ratio * len(self.open_squares))
        self.window = None

    def _observation(self):
        obs = np.zeros((self.obs_n,), dtype = np.float32)
        if self._is_open(self.agent[0], self.agent[1] - 1):
            obs[0] += 1.0
        if self._is_open(self.agent[0] + 1, self.agent[1]):
            obs[1] += 1.0
        if self._is_open(self.agent[0], self.agent[1] + 1):
            obs[2] += 1.0
        if self._is_open(self.agent[0] - 1, self.agent[1]):
            obs[3] += 1.0
        
        if self.include_last_action_and_reward and self.last_action:
            obs[self.last_action + 4] = 1.0
            obs[8] = self.last_reward
            
        return obs

    def step(self, action):
        # 0 = up, 1 = right, 2 = down, 3 = left
        assert action in [0,1,2,3], 'Action invalid for Grid Maze.'
        self.n_steps += 1
        self.last_action = action
        info = {}
        new_loc = list(self.agent)
        if action == 0:
            new_loc[1] -= 1
        elif action == 1:
            new_loc[0] += 1
        elif action == 2:
            new_loc[1] += 1
        else:
            new_loc[0] -= 1
        new_loc = tuple(new_loc)

        info['correct_direction'] = new_loc in self._neighbors(*self.agent)
        if info['correct_direction']:
            self.agent = new_loc
        done = self.agent == self.end_node
        rew = 1.0 if done else -.01 # TODO: add different types of reward?
        if not info['correct_direction']:
            rew -= .01
        self.last_reward = rew

        if self.n_steps >= self.max_allowed_steps:
            done = True
            info['truncated'] = True
        else:
            info['truncated'] = False

        return self._observation(), rew, done, info

    def reset(self):
        self.end_node, self.agent = random.sample(self.open_squares, 2)
        self.n_steps = 0
        self.last_action = None
        return self._observation()

    def render(self):
        if not self.window:
            self.window = tkinter.Tk()
            self.canvas = tkinter.Canvas(self.window, width = self.size_x * self.grid_pixels,
                                                      height = self.size_y * self.grid_pixels)
            self.canvas.pack()
        for x,y in itertools.product(range(self.size_x), range(self.size_y)):
            color = 'black'
            x1, y1 = x * self.grid_pixels, y * self.grid_pixels
            x2, y2 = x1 + self.grid_pixels, y1 + self.grid_pixels
            if (x,y) == self.agent:
                color = 'blue'
                x1 += self.grid_pixels / 8
                y1 += self.grid_pixels / 8
                x2 -= self.grid_pixels / 8
                y2 -= self.grid_pixels / 8
            elif (x,y) == self.end_node:
                color = 'green'
            elif self.grid[x][y]:
                color = 'white'
            self.canvas.create_rectangle(x1, y1, x2, y2, fill = color)
        self.window.update()
        time.sleep(0.05) # allows us to see what is happening

    def close(self):
        if self.window:
            self.window.destroy()

if __name__ == '__main__':
    maze = GridExplore(6, 6)
    obs = maze.reset()
    print("initial observation:", obs)
    done = False
    maze.render()
    while not done:
        obs, rew, done, info = maze.step(maze.action_space.sample())
        maze.render()
        print("*" * 64)
        print("observation:", obs)
        print("reward:", rew)
        print("info", info)
