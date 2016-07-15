import gym
import os
from scipy import misc
import numpy as np
from collections import deque

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
        if name in {'breakout', 'pong'}:
            self.action_space = [2, 3]
        elif name == 'gridworld':
            pass
        else:
            self.action_space = self.env.action_space

        self.resize = tuple(grid_size)
        self.history_length = last_n
        self.history = deque(maxlen=last_n)
        self.old_preprocessed_screen = []
        self.delta_preprocessing = delta_preprocessing

    def reset(self):
        """
        This should set the initial state
        """
        observation = self.env.reset()
        preprocessed_observation = self.preprocess(observation)
        self.history.clear()
        if self.delta_preprocessing:
            observation, reward, done, info = self.env.step(np.random.choice(self.action_space))
            preprocessed_observation = self.preprocess(observation)

        for _ in xrange(self.history_length):
            self.history.append(preprocessed_observation)
        state = [str(frame_num) + "_" + str(pixel) for frame_num, frame in enumerate(self.history) for pixel in frame]

        return state

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
        if len(self.history) == self.history_length:
            observation, reward, done, info = self.env.step(action)
            preprocessed_observation = self.preprocess(observation)
            self.history.append(preprocessed_observation)

        elif len(self.history) < self.history_length:
            while len(self.history) < self.history_length:
                observation, reward, done, info = self.env.step(action)
                preprocessed_observation = self.preprocess(observation)
                self.history.append(preprocessed_observation)

        state = [str(frame_num) + "_" + str(pixel) for frame_num, frame in enumerate(self.history) for pixel in frame]
        # state = np.hstack(self.history)
        return state, reward, done, info

    def render(self):
        self.env.render()

    # --------- Methods for Atari Games --------- #
    def preprocess(self, observation):
        if self.name in {'breakout'}:
            t1 = observation[95:195].mean(axis=2)[::3, ::3]
            t1[t1 == 142] = 0 # Kill border
            t1[t1 == 118] = 0 # Kill background

        elif self.name == 'pong':
            t1 = observation[35:195]  # crop
            t1 = t1[::3, ::3, 0]  # downsample by factor of 3
            t1[t1 == 144] = 0  # erase background (background type 1)
            t1[t1 == 109] = 0  # erase background (background type 2)

        else:
            # crop, grayscale and downsample
            t1 = misc.imresize(observation[35:195].mean(axis=2), self.resize, interp='bilinear')

        if self.delta_preprocessing:
            if len(self.old_preprocessed_screen) == 0:
                preprocessed = []
            else:
                resized_grayscale_obs = t1 - self.old_preprocessed_screen
                # Return only non-zero pixels for binary sparse features
                preprocessed = np.nonzero(resized_grayscale_obs.ravel())[0]
            self.old_preprocessed_screen = t1

        else:
            preprocessed = np.nonzero(t1.ravel())[0]

        return preprocessed

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