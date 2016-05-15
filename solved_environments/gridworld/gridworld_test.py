from environments.gridworld import GridWorld
import rl_learning as rl
from rl_learning import RLAgent
import cPickle as pickle
from bandits import BanditAlgorithm


def test_training_TD_for_gridworld(model_class, epochs, train=True):
    gridworld = GridWorld()
    if train:
        policy, model = rl.train_reinforcement_strategy_temporal_difference(epochs=epochs, game_obs=gridworld, model_class=model_class)
    random_stat, model_stat = rl.test_policy_with_random_play(gridworld)
    return random_stat, model_stat

    # Record MSE for each epoch may be?
    # Record % of wins

# random_stat, model_stat = test_training_TD_for_gridworld(model_class='vw_python', epochs=50000, train=True)
# print random_stat
# print model_stat



def train_rl_agent(env_name, train=True):

    if env_name == 'gridworld':
        from environments.gridworld import GridWorld
        env = GridWorld()

    rl_params = {'epochs': 50000, 'experience_replay_size': 20, 'batchsize': 10, 'gamma': 0.1, 'skip_frames': 1}
    model_params = {'class': 'vw_python', 'base_folder_name': env.base_folder_name, 'loss_function': 'squared',
                    'l2': 0.000000001, 'lrq': 'sdsd200', 'b': 20}
    bandit_params = 0.9

    # Initialize RL agent
    rl_agent = RLAgent(epochs=rl_params['epochs'], experience_replay_size=rl_params['experience_replay_size'], batchsize=rl_params['batchsize'],
                       gamma=rl_params['gamma'], skip_frames=rl_params['skip_frames'])
    if train:
        model, bandit_algorithm = rl_agent.initialize(model_params, bandit_params)
        rl_agent.train_q_function(env, model, bandit_algorithm)

    else:
        #random_stat, model_stat = rl.test_policy_with_random_play(env)
        model, bandit_algorithm = rl_agent.initialize(model_params, bandit_params, test=True)
        random_stat, model_stat = rl_agent.test_q_function(env, model, bandit_algorithm)
        print random_stat
        print model_stat

#train_rl_agent(env_name='gridworld', train=True)
train_rl_agent(env_name='gridworld', train=False)