from abc import ABCMeta, abstractmethod
import os
from scipy import misc
from collections import deque

class AbstractGame(object):

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.base_folder_name = os.path.dirname(os.path.realpath(__file__)).replace('environments', 'solved_environments') + '/' + name
        # state should be a vector (1d array)

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
    def step(self, action):
        """
        This should update state after interacting with the
        environment. Mostly used for temporal difference learning
        :return: <observation, reward, done, info>
        """
        pass

    # --------- Methods for Atari Games --------- #
    def preprocess_screen(self, original_screen, size=None):
        """
        :param size:
        :return:
        """
        grayscale_screen = self._convert_rgb_to_grayscale(original_screen)
        if size:
            grayscale_screen = self._resize_image(grayscale_screen, size)

        return grayscale_screen

    def _convert_rgb_to_grayscale(rgb_image):
        """
        Convert rgb image to to grayscale
        (assuming rgb_image is ndarray)
        :return:
        """
        grayscale_image = rgb_image.mean(axis=2)
        # grayscale_image = np.ma.average(rgb_image, axis=2, weights=[0.299, 0.587 ,0.114]) # Weighted average
        return grayscale_image

    def _resize_image(self, original_image, size):
        """
        Resize imgae
        :param original_image:
        :param size:
        :return:
        """
        return misc.imresize(original_image, size)