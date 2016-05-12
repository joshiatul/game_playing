from abc import ABCMeta, abstractmethod
import os

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