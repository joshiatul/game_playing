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
    def preprocess_screen(self, original_screen, size=None):
        """
        :param size:
        :return:
        """
        grayscale_screen = self._convert_rgb_to_grayscale(original_screen)
        if size:
            grayscale_screen = self._resize_image(grayscale_screen, size)

        # Also return only non-zero pixels of the screen
        grayscale_screen = grayscale_screen.flatten()
        non_zero_pixels_of_grayscale_screen = np.nonzero(grayscale_screen)[0]

        return non_zero_pixels_of_grayscale_screen

    def _convert_rgb_to_grayscale(self, rgb_image):
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

    def preprocess(self, observation):
        # crop
        observation = observation[35:195]
        # grayscale
        grayscale_obs = observation.mean(axis=2)
        # resize
        resized_grayscale_obs = misc.imresize(grayscale_obs, (80, 80))
        # new-old
        if len(self.old_preprocessed_screen) > 0:
            screen_delta = resized_grayscale_obs - self.old_preprocessed_screen
            # Store as old screen
            self.old_preprocessed_screen = resized_grayscale_obs
            # flatten
            screen_delta = screen_delta.flatten()
            # return non-zero pixels as preprocessed screen
            screen_delta = np.nonzero(screen_delta)[0]
            screen_delta = [str(i) for i in screen_delta]
            return screen_delta

        else:
            # Store as old screen
            self.old_preprocessed_screen = resized_grayscale_obs
            return


