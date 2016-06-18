import gym
import os
from scipy import misc
import numpy as np
from collections import deque

class Environment(object):
    def __init__(self, name, width=None, height=None, last_n=None, delta_preprocessing=False):
        # self.base_folder_name = os.path.dirname(os.path.realpath(__file__)).replace('environments', 'solved_environments') + '/' + name
        # # TODO simplfy for all atari games
        if name == 'breakout':
            self.env = gym.make('Breakout-v0')
        elif name == 'pong':
            self.env = gym.make('Pong-v0')

        # gym returns 6 possible actions for breakout and pong.
        # I think only 3 are used for both. So making life easier
        # with "LEFT", "RIGHT", "NOOP" actions space.
        if name in {'breakout', 'pong'}:
            self.action_space = [1, 2, 3]

        self.resize = (width, height)
        self.history_length = last_n
        self.history = deque(maxlen=last_n)
        self.old_preprocessed_screen = []
        self.delta_preprocessing = delta_preprocessing

    def reset(self):
        """
        This should set the initial state
        """
        observation = self.env.reset()
        current_state = self.preprocess(observation)
        self.history.clear()
        # for _ in xrange(self.history_length):
        #     if not self.delta_preprocessing:
        #         self.history.appendleft(preprocessed_observation)
        return current_state

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
            self.history.appendleft(preprocessed_observation)

        elif len(self.history) < self.history_length:
            while len(self.history) < self.history_length:
                observation, reward, done, info = self.env.step(action)
                preprocessed_observation = self.preprocess(observation)
                self.history.appendleft(preprocessed_observation)

        state = np.hstack(self.history)
        return state, reward, done, info

    def render(self):
        self.env.render()

    # --------- Methods for Atari Games --------- #
    def preprocess(self, observation):
        # crop to capture playing area (for breakout and pong)
        observation = observation[35:195]
        # grayscale
        # I = I[::2,::2,0] # downsample by factor of 2 and kill rgb
        grayscale_obs = observation.mean(axis=2)
        # resize
        resized_grayscale_obs = misc.imresize(grayscale_obs, self.resize)

        preprocessed = resized_grayscale_obs.ravel()

        if self.delta_preprocessing and len(self.old_preprocessed_screen) > 0:
            resized_grayscale_obs = preprocessed - self.old_preprocessed_screen
            # Return only non-zero pixels for binary sparse features
            resized_grayscale_obs = np.nonzero(resized_grayscale_obs.ravel())[0]

            # Stringyfy
            resized_grayscale_obs = [str(i) for i in resized_grayscale_obs]
        else:
            resized_grayscale_obs = []

        self.old_preprocessed_screen = preprocessed
        return resized_grayscale_obs

    def clip_reward(self, reward, done):
        if reward > 0:
            reward = 20
        elif reward <= 0 and done:
            reward = -20
        elif reward <= 0 and not done:
            reward = -1
        return reward