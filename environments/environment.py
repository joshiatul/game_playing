import gym
import os
from scipy import misc
import numpy as np
from collections import deque
import mmh3

class Environment(object):
    def __init__(self, name, grid_size=None, last_n=None, delta_preprocessing=False):
        # self.base_folder_name = os.path.dirname(os.path.realpath(__file__)).replace('environments', 'solved_environments') + '/' + name
        # # TODO simplfy for all atari games
        self.name = name
        if name == 'breakout':
            self.env = gym.make('Breakout-v0')
        elif name == 'pong':
            self.env = gym.make('Pong-v0')
        elif name == 'gridworld':
            pass
        else:
            self.env = gym.make(name)

        # gym returns 6 possible actions for breakout and pong.
        # I think only 3 are used for both. So making life easier
        # with "LEFT", "RIGHT", "NOOP" actions space.
        # env.unwrapped.get_action_meanings()
        if name in {'breakout', 'pong'}:
            self.action_space = [2, 3]
        elif name == 'gridworld':
            pass
        else:
            self.action_space = self.env.action_space

        self.resize = tuple(grid_size)
        self.history_length = last_n
        self.history = deque(maxlen=last_n)
        self.prev_x = None
        self.delta_preprocessing = delta_preprocessing

    def reset(self):
        """
        This should set the initial state
        """
        observation = self.env.reset()
        self.prev_x = None

        return observation

    def complete_one_episode(self):
        """
        Used for Monte-Carlo learning
        """
        pass

    def step(self, action):
        """
        This should update state after interacting with the
        environment. Mostly used for temporal difference learning
        :return: <observation, reward, done, info>
        """
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def render(self):
        self.env.render()

    # --------- Methods for Atari Games --------- #
    def preprocess(self, observation):
        if self.name in {'breakout'}:
            t1 = observation[95:195].mean(axis=2)[::2, ::2]
            t1[t1 == 142] = 0 # Kill border
            t1[t1 == 118] = 0 # Kill background
            t1[t1 != 0] = 1  # everything else (paddles, ball) just set to 1
            return t1.astype(np.float).ravel()

        elif self.name == 'pong':
            t1 = observation[35:195]  # crop
            t1 = t1[::2, ::2, 0]  # downsample by factor of 2
            t1[t1 == 144] = 0  # erase background (background type 1)
            t1[t1 == 109] = 0  # erase background (background type 2)
            t1[t1 != 0] = 1  # everything else (paddles, ball) just set to 1
            return t1.astype(np.float).ravel()

    def sparsify(self, x):
        sparse_x = np.nonzero(x)[0]
        tag = str(mmh3.hash128("_".join('pix_' + str(i) for i in sparse_x)))
        state = " |state " + " ".join('pix_' + str(i) for i in sparse_x)  + " tag_" + tag
        return state

    def preprocess_and_sparsify(self, observation):
        cur_x = self.preprocess(observation)
        if self.delta_preprocessing:
            x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(54*54)
            self.prev_x = np.copy(cur_x)
            sparse_x = self.sparsify(x)
        else:
            sparse_x = self.sparsify(cur_x)
        return sparse_x

    def clip_reward(self, reward, done):
        # Do not clip reward for gridworld
        if self.name == 'gridworld':
            return reward
        else:
            if reward > 0:
                reward = 10
            elif reward < 0:
                reward = -10
        return reward