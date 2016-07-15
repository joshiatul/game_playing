import random
from collections import namedtuple
import environment
import numpy as np
import pandas as pd
import math


def flatten_list_of_lists(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


class GridWorld(environment.Environment):
    def __init__(self, grid_size):

        super(GridWorld, self).__init__(name='gridworld', grid_size=grid_size)
        self.action_space = ['up', 'down', 'left', 'right']
        self.all_decisions = ['up', 'down', 'left', 'right']

        game_objects = ['player', 'wall', 'pit', 'win']
        game_state_tuples = [[obs + '_x', obs + '_y'] for obs in game_objects]
        coordinate_list = flatten_list_of_lists(game_state_tuples)
        global state_info
        state_info = namedtuple('state_info', coordinate_list)
        self.state_info = state_info
        global coordinates
        coordinates = namedtuple('coordinates', ['x', 'y'])
        self.coordinates = coordinates
        self.all_used_coordinates = {'x': set(), 'y': set()}
        self.size = grid_size[0] # Assuming square grid for now


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
            # Only 4 objects on the screen for now!
            random_coors = [self.coordinates(i, j) for i, j in zip(random.sample(xrange(0, self.size), self.size), [random.randint(0, self.size-1) for _ in xrange(0, 4)])]
            self.player_info, self.wall_info, self.pit_info, self.win_info = random_coors


        # Else generate only player and win randomly
        else:
            x1, y1 = 2, 1
            self.wall_info = self.coordinates(x1, y1)
            x2, y2 = 1, 0
            self.pit_info = self.coordinates(x2, y2)
            x1, y1 = 3, 0
            self.player_info = self.coordinates(x1, y1)
            x2, y2 = 0, 1
            self.win_info = self.coordinates(x2, y2)

        game_state = (self.player_info, self.wall_info, self.pit_info, self.win_info)
        game_state = tuple(('feature' + str(idx) + '-' + '-'.join(str(x) for x in obs) for idx, obs in enumerate(game_state)))
        self.state = game_state
        self.game_status = 'in process'
        return game_state

    def render(self):
        grid = np.zeros((self.size, self.size), dtype='<U2')

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
            new_loc = self.coordinates(self.player_info.x, self.player_info.y - 1)
            if new_loc != self.wall_info and new_loc.y >= 0:
                self.player_info = new_loc

        elif action == 'right':
            new_loc = self.coordinates(self.player_info.x, self.player_info.y + 1)
            if new_loc != self.wall_info and new_loc.y <= self.size-1:
                self.player_info = new_loc

        elif action == 'up':
            new_loc = self.coordinates(self.player_info.x - 1, self.player_info.y)
            if new_loc != self.wall_info and new_loc.x >= 0:
                self.player_info = new_loc

        elif action == 'down':
            new_loc = self.coordinates(self.player_info.x + 1, self.player_info.y)
            if new_loc != self.wall_info and new_loc.x <= self.size-1:
                self.player_info = new_loc

        # Reset state
        game_state = (self.player_info, self.wall_info, self.pit_info, self.win_info)
        game_state = tuple(('feature' + str(idx) + '-' + '-'.join(str(x) for x in obs) for idx, obs in enumerate(game_state)))

        self.state = game_state

        # Get and return reward
        reward = self.get_reward()
        # Needs to return <observation, reward, done, info>
        done = True if self.game_status != 'in process' else False
        return self.state, reward, done, []

    def get_reward(self):
        if self.player_info == self.pit_info:
            self.game_status = 'player loses'
            return -10
        elif self.player_info == self.win_info:
            self.game_status = 'player wins'
            return 10
        else:
            return -1

    def preprocess(self, observation):
        # No preprocessing for gridworld
        return observation

if __name__ == "__main__":
    gridworld = GridWorld()
    gridworld.reset()
    print gridworld.player_info
    gridworld.render()
    # reward = gridworld.step('down')
    # print gridworld.player_info
    # gridworld.render()
    # reward = gridworld.step('down')
    # print gridworld.player_info
    # print gridworld.render()
    # reward = gridworld.step('down')
    # print gridworld.player_info
    # print gridworld.render()
    # reward = gridworld.step('right')
    # print gridworld.player_info
    # print gridworld.render()
    #
    # print reward
