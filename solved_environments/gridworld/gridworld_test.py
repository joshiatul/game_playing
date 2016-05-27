from environments.gridworld import GridWorld
import rl_learning as rl
from rl_learning import RLAgent
import cPickle as pickle
from bandits import BanditAlgorithm


def train_rl_agent(env_name, train=True):

    if env_name == 'gridworld':
        from environments.gridworld import GridWorld
        env = GridWorld()

    rl_params = {'epochs': 60000, 'experience_replay_size': 200, 'batchsize': 20, 'gamma': 0.1, 'skip_frames': 1, 'max_steps': 30}
    model_params = {'class': 'vw_python', 'base_folder_name': env.base_folder_name, 'loss_function': 'squared',
                    'l2': 0.0000000001, 'lrq': 'sdsd200', 'b': 20, 'l': 0.5}
    bandit_params = 0.9

    # Initialize RL agent
    rl_agent = RLAgent(epochs=rl_params['epochs'], experience_replay_size=rl_params['experience_replay_size'], batchsize=rl_params['batchsize'],
                       gamma=rl_params['gamma'], skip_frames=rl_params['skip_frames'], max_steps=rl_params['max_steps'])
    if train:
        model, bandit_algorithm = rl_agent.initialize(model_params, bandit_params)
        rl_agent.train_q_function(env, model, bandit_algorithm)

    else:
        #random_stat, model_stat = rl.test_policy_with_random_play(env)
        model, bandit_algorithm = rl_agent.initialize(model_params, bandit_params, test=True)
        random_stat, model_stat = rl_agent.test_q_function(env, model, bandit_algorithm)
        print random_stat
        print model_stat

train_rl_agent(env_name='gridworld', train=True)
train_rl_agent(env_name='gridworld', train=False)