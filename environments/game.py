from abc import ABCMeta, abstractmethod
import os
from scipy import misc
import numpy as np

class AbstractGame(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.base_folder_name = os.path.dirname(os.path.realpath(__file__)).replace('environments', 'solved_environments') + '/' + name
        self.old_preprocessed_screen = []

    @abstractmethod
    def reset(self):
        """
        This should set the initial state
        """
        pass

    @abstractmethod
    def complete_one_episode(self):
        """
        Used for Monte-Carlo learning
        """
        pass

    @abstractmethod
    def step(self, action, skip_frames=0):
        """
        This should update state after interacting with the
        environment. Mostly used for temporal difference learning
        :return: <observation, reward, done, info>
        """
        pass

    # --------- Methods for Atari Games --------- #
    def preprocess(self, observation):
        # crop to capture playing area
        observation = observation[35:195]
        # grayscale
        # I = I[::2,::2,0] # downsample by factor of 2 and kill rgb
        grayscale_obs = observation.mean(axis=2)
        # resize
        resized_grayscale_obs = misc.imresize(grayscale_obs, (40, 40))
        # new-old
        if len(self.old_preprocessed_screen) > 0:
            screen_delta = resized_grayscale_obs - self.old_preprocessed_screen
            # Store as old screen
            self.old_preprocessed_screen = resized_grayscale_obs
            # flatten
            screen_delta = screen_delta.ravel()
            # return non-zero pixels as preprocessed screen
            screen_delta = np.nonzero(screen_delta)[0]
            screen_delta = [str(i) for i in screen_delta]
            return screen_delta

        else:
            # Store as old screen
            self.old_preprocessed_screen = resized_grayscale_obs
            return


    def clip_reward(self, reward, done):
        if reward > 0:
            reward = 20
        elif reward <= 0 and done:
            reward = -20
        elif reward <= 0 and not done:
            reward = -1
        return reward