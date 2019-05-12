import gym
import random
import numpy as np

class PermEnv(gym.Env):
    """
    Chooses a random permutation of k elements and cycles through each time step.
    Action space is 'k'. Observation is previous reward and action.
    Reward is 1.0 if action is the "current" element
    and 0.0 otherwise.
    """
    def __init__(self, k, max_runs):
        self.order = list(range(k))
        self.k = k
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape = [k + 1])
        self.max_runs = max_runs

    def _observation(self):
        obs = np.zeros(self.k + 1)
        if self.prev_act != -1:
            obs[self.prev_act] = 1.0
        obs[self.k] = self.prev_rew
        return obs

    def step(self, action):
        rew = (action == self.order[self.t % self.k])
        self.t += 1
        self.prev_rew = rew
        self.prev_act = action
        obs = self._observation()
        done = (self.t >= self.max_runs)

        return obs, rew, done, {}

    def reset(self):
        self.t = 0
        self.prev_rew = 0.0
        self.prev_act = -1
        random.shuffle(self.order)
        return self._observation()

    def render(self):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    pe = PermEnv(4, 16)
    obs = pe.reset()
    while True:
        print(obs)
        action = int(input())
        obs, rew, done, info = pe.step(action)
        if done:
            break