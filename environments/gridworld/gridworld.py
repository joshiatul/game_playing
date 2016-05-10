import random
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from .. import game
import os
import numpy as np
import pandas as pd
import math


def flatten_list_of_lists(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

class GridWorld(game.AbstractGame):

    def __init__(self):
        self.base_folder_name = os.path.dirname(os.path.realpath(__file__))
        self.action_space = ['up', 'down', 'left', 'right']
        self.all_decisions = ['up', 'down', 'left', 'right']

        game_objects = ['player', 'wall', 'pit', 'win']
        game_state_tuples = [[obs + '_x', obs + '_y'] for obs in game_objects]
        coordinate_list = flatten_list_of_lists(game_state_tuples)
        self.state_info = namedtuple('state_info', coordinate_list)

        self.coordinates = namedtuple('coordinates', ['x', 'y'])
        self.all_used_coordinates = {'x': set(), 'y': set()}

        self.game_status = None
        self.state = None

    def random_unsed_coordinates(self, a, b):
        # TODO Fix this
        xa = np.random.randint(a, b)
        while xa not in self.all_used_coordinates['x']:
            ya = np.random.randint(a, b)
            while ya not in self.all_used_coordinates['y']:
                self.all_used_coordinates['x'].add(xa)
                self.all_used_coordinates['y'].add(ya)
                return xa, ya

    def evaluate_and_modify_possible_actions(self):
        """
        Based on current state evaluate what actions are possible
        for a player
        All walls block player movement
        This needs to be evaluated after every step
        """
        remove_actions = set()
        # pl_x, pl_y = self.state[0]
        # wall_x, wall_y = self.state[1]
        #
        # # If player is near wall (grid boundary or wall object) action is blocked
        # if pl_y == 0 or pl_y-1 == wall_y: remove_actions.add('left')
        # elif pl_y == 3 or pl_y+1 == wall_y: remove_actions.add('right')
        #
        # if pl_x == 0 or pl_x-1 == wall_x: remove_actions.add('up')
        # elif pl_x == 3 or pl_x+1 == wall_x: remove_actions.add('down')

        self.action_space = [i for i in self.all_decisions if i not in remove_actions]


    def reset(self, full_rnd=True):
        """
        What exactly we know about the game beforehand?
         - all possible actions we can take
         - and image (we don't know player, wall, pit or goal)
         - game's internal environment returns reward based on the movement and that's how we should learn the rules
         - so our design matrix should only be set of pixels (or image representation) AND decision taken
         - all we know is : there are 4 things on the screen.. lets use sparse representation
        """

        if full_rnd:
            random_coors = [self.coordinates(i,j) for i,j in zip(random.sample(xrange(0,4), 4), [random.randint(0,3) for _ in xrange(0, 4)])]
            self.player_info, self.wall_info, self.pit_info, self.win_info = random_coors

        # Else generate only player and win randomly
        else:
            x1, y1 = 1,3
            self.wall_info = self.coordinates(x1, y1)
            x2, y2 = 2,3
            self.pit_info = self.coordinates(x2, y2)

            x3, y3 = (random.randint(0,3), random.randint(0,3))
            while (x1, y1) == (x3, y3):
                x3, y3 = (random.randint(0,3), random.randint(0,3))
            self.player_info = self.coordinates(x3, y3)

            x4, y4 = (random.randint(0,3), random.randint(0,3))
            while (x2, y2) == (x4, y4):
                x4, y4 = (random.randint(0,3), random.randint(0,3))
            self.win_info = self.coordinates(x4, y4)

        game_state = (self.player_info, self.wall_info, self.pit_info, self.win_info)
        self.state = game_state
        self.game_status = 'in process'
        # Certain actions are not possible if the player is sitting on the edge
        self.evaluate_and_modify_possible_actions()

    def display_grid(self):
        grid = np.zeros((4,4), dtype='<U2')

        grid[self.player_info.x, self.player_info.y] = 'P'
        grid[self.wall_info.x, self.wall_info.y] = 'W'
        if self.player_info != self.pit_info:
            grid[self.pit_info.x, self.pit_info.y] = '-'

        if self.player_info != self.win_info:
            grid[self.win_info.x, self.win_info.y] = '+'

        print pd.DataFrame(grid)


    def complete_one_episode(self):
        pass

    def step(self, action):
        self.player_old_state = self.player_info
        if action == 'left':
            new_loc = self.coordinates(self.player_info.x, self.player_info.y-1)
            if new_loc != self.wall_info and new_loc.y >= 0:
                self.player_info = new_loc

        elif action == 'right':
            new_loc = self.coordinates(self.player_info.x, self.player_info.y+1)
            if new_loc != self.wall_info and new_loc.y <= 3:
                self.player_info = new_loc

        elif action == 'up':
            new_loc = self.coordinates(self.player_info.x-1, self.player_info.y)
            if new_loc != self.wall_info and new_loc.x >= 0:
                self.player_info = new_loc

        elif action == 'down':
            new_loc = self.coordinates(self.player_info.x+1, self.player_info.y)
            if new_loc != self.wall_info and new_loc.x <= 3:
                self.player_info = new_loc

        # Reset state
        game_state = (self.player_info, self.wall_info, self.pit_info, self.win_info)
        # Certain actions are not possible if the player is sitting on the edge
        self.evaluate_and_modify_possible_actions()
        self.state = game_state

        # Get and return reward
        reward = self.get_reward()
        return reward

    def get_reward(self):
        if self.player_info == self.pit_info:
            self.game_status = 'player loses'
            return -20
        elif self.player_info == self.win_info:
            self.game_status = 'player wins'
            return 20
        elif self.player_info == self.player_old_state:
            return -10
        else:
            # Return distance from win (player looks at screen so i think this is fare)
            return -(math.sqrt((self.player_info.x - self.win_info.x)**2 + (self.player_info.y - self.win_info.y)**2))
            #return -1


if __name__ == "__main__":
    gridworld = GridWorld()
    gridworld.reset()
    print gridworld.player_info
    gridworld.display_grid()
    reward = gridworld.step('down')
    print gridworld.player_info
    gridworld.display_grid()
    reward = gridworld.step('down')
    print gridworld.player_info
    print gridworld.display_grid()
    reward = gridworld.step('down')
    print gridworld.player_info
    print gridworld.display_grid()
    reward = gridworld.step('right')
    print gridworld.player_info
    print gridworld.display_grid()

    print reward