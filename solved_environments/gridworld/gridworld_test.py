import rl_learning as rl

def train_rl_agent(env_name, train=True):

    memory_structure = 'asynchronus_methods' #''experience_replay'
    memory_structure_params = {'actor_critic_method': False}
    # memory_structure_params = {'experience_replay_size': 40, 'batchsize': 20, 'minibatch_method': 'random'}
    rl_params = {'memory_structure': memory_structure, 'memory_structure_params': memory_structure_params, 'gamma': 0.99, 'max_steps': 30,
                 'n_steps_ahead': 3}
    model_params = {'class': 'vw_python', 'loss_function': 'squared',
                    'l2': 0.0000000001, 'lrq': 'se7', 'b': 20, 'l': 1}
    bandit_params = {'start_epsilon': 0.9, 'anneal_epsilon_timesteps': 4000}

    if train:
        rl.train_with_threads(env_name, rl_params, bandit_params, model_params, num_of_threads=8, epochs=10000, train=True,
                              display_state=False, use_processes=False)
    else:
        stat = rl.test_trained_model_with_random_play(env_name, test_games=100, render=False)
        print stat


train_rl_agent(env_name='gridworld', train=True)
train_rl_agent(env_name='gridworld', train=False)

# With experience-replay
# With single thread 10k epochs not enough, test accuracy = 66% (88 sec)
# With 4 threads test accuracy = 97% (287 sec, although gridworld is already CPU heavy)

# Without experience-replay
# 16 threads 10 k epochs, test accuracy = 95% (108 sec) with threads (processes work but slow)
# with gridsize 5x5 still getting only 75% wins with 20 k epochs (about 400 sec)
# quantile loss function works fine as well (91%, 95sec)