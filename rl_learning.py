from model import Model
import bandits
import random
import time
import cPickle as pickle
from collections import Counter, deque
import numpy as np
import os
from environments.environment import Environment
import threading
import multiprocessing
from multiprocessing.managers import BaseManager

# http://stackoverflow.com/questions/26499548/accessing-an-attribute-of-a-multiprocessing-proxy-of-a-class
# http://stackoverflow.com/questions/28612412/how-can-i-share-a-class-between-processes-in-python


def learn_Q_function(all_observed_decision_states, reward, model):
    """
    Episodic learning (mostly for lookup table method) - helper method
    """
    if model.model_class == 'lookup_table':
        model.fit(all_observed_decision_states, reward)

    elif model.model_class == 'scikit' or model.model_class == 'vw' or model.model_class == 'vw_python':
        for decision_state in all_observed_decision_states:
            X_new, y_new = model.return_design_matrix(decision_state, reward)
            model.X.append(X_new)
            model.y.append(y_new)

        if model.buffer == 1000:
            model.fit(model.X, model.y)

            # TODO Instead of killing entire buffer we can keep a few and kill only the subset
            model.clean_buffer()

    return model


def train_reinforcement_learning_strategy(num_sims=1, game_obs='blackjack', model_class='lookup_table'):
    """
    Episodic learning (mostly for lookup table method)
    """
    start_time = time.time()
    # Initialize model
    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    banditAlgorithm = BanditAlgorithm(params=0.1)
    model.initialize()

    model.all_possible_decisions = game_obs.action_space

    for _ in xrange(num_sims):
        model.buffer += 1

        # Initialize game
        game_obs.reset()
        if game_obs.game_status != 'in process':
            continue

        all_observed_decision_states, reward = game_obs.complete_one_episode(banditAlgorithm, model)
        model = learn_Q_function(all_observed_decision_states, reward, model)

    model.finish()
    elapsed_time = int(time.time() - start_time)
    print ": took time:" + str(elapsed_time)
    return banditAlgorithm.policy, model

## ---------------------------------- Actor-Critic / n-step TD learning -------------------------------------------------------------------------- ##
    # TODO Not using td lambda implementation for now (i assume if i do this, i can get away with less epochs)
    # def append_design_matrix_with_td_lambda(self, episode_er, move_er, reward_er, lamb):
    #     # Do eligibility traces only if reward is substantial
    #     if reward_er >= abs(-2) and lamb > 0:
    #         for backstep in reversed(xrange(move_er)):
    #             weight_er = (1 - lamb) * (lamb ** (move_er - backstep))
    #             if weight_er < 0.001: break # Break if weight is too low
    #             # TODO Implement frequency based trace, for now doing only recency based trail
    #             key1 = (episode_er, backstep)
    #             if key1 in self.experience_replay_obs.experience_replay:
    #                 old_state_erl, action_erl, reward_erl, new_state_erl, is_terminal, episode_erl, move_erl = self.experience_replay_obs.experience_replay[key1]
    #                 reward_erl += self.gamma * reward_er
    #                 reward_er = reward_erl
    #
    #                 X_new, y_new = self.model.return_design_matrix((old_state_erl, action_erl), reward_erl, weight_er)
    #                 self.model.X.append(X_new)
    #                 self.model.y.append(y_new)

def play_with_environment(environment_params, model, statistics, rl_params, bandit_params, epochs, thread_id=1, train=True, display_state=False):
    """
    Simple temporal difference learning
    with experience-replay
    :return:
    """
    env = make_environment(environment_params)
    epsilon = bandit_params.get('start_epsilon', 0.9)
    end_epsilon = bandits.sample_end_epsilon()
    model_class = 'random' if not model else model.return_model_class()
    actor_critic_method = rl_params.get('memory_structure_params', {}).get('actor_critic_method', False)
    X, y, X_critic, y_critic = [], [], [], []

    if train and rl_params.get('memory_structure', 'asynchronus_methods') == 'experience_replay':
        experience_replay_obs = ExperienceReplay(type='deque', batchsize=rl_params['memory_structure_params']['batchsize'],
                                                 experience_replay_size=rl_params['memory_structure_params']['experience_replay_size'],
                                                 minibatch_method=rl_params['memory_structure_params']['minibatch_method'])

    if train:
        print "------ Starting thread: " + str(thread_id) + " with final epsilon ", end_epsilon
        time.sleep(3 * thread_id)

    for episode in xrange(1, epochs + 1):

        if not train and display_state: print "Game #-----: " + str(episode)

        # Initialize game and per episode counters
        current_state = env.reset()
        total_episodic_reward = 0
        episodic_max_q = 0

        if train and model.if_exists() and not actor_critic_method:
            epsilon = bandits.decrement_epsilon(epochs, epsilon, bandit_params.get('anneal_epsilon_timesteps', 10000), end_epsilon)

        # Start playing the game
        for move in xrange(1, rl_params['max_steps'] + 1):

            if display_state: env.render()
            episodic_rewards = []
            states = []

            # Look n step ahead
            for _ in xrange(rl_params.get('n_steps_ahead', 1)):
                # Figure out best action based on policy
                if not actor_critic_method:
                    action, max_q_value = bandits.select_action_with_epsilon_greedy_policy(current_state, env.action_space, model,
                                                                                                        epsilon=epsilon, test=not train)
                else:
                    action, max_q_value = bandits.return_action_based_on_softmax_policy(current_state, model, env.action_space)

                # Take step / observe reward / preprocess / update counters
                if not train and display_state: print "Taking action: #-----: " + str(action)
                new_state, reward, done, info = env.step(action)
                clipped_reward = env.clip_reward(reward, done)
                episodic_rewards.append(clipped_reward)
                states.append(current_state)
                total_episodic_reward += reward
                td_error = clipped_reward - max_q_value
                episodic_max_q += max_q_value

                # Update state
                current_state = new_state

                if done:
                    break

            if train:
                bootstrapped_reward = return_bootstrapped_reward(env, model, rl_params['gamma'], new_state, done, actor_critic_method)

                for i in xrange(len(episodic_rewards)-1, -1, -1):
                    bootstrapped_reward = episodic_rewards[i] + rl_params['gamma'] * bootstrapped_reward

                    if rl_params['memory_structure'] == 'experience_replay':
                        X, y = generate_training_samples_with_experience_replay(experience_replay_obs, env, model, X, y, episode_key=(episode, move),
                                                                                gamma=rl_params['gamma'],
                                                                                state_tuple=(
                                                                                states[i], action, bootstrapped_reward, new_state, done, episode,
                                                                                move, td_error))

                    else:
                        X_new, y_new = model.return_design_matrix((states[i], action), bootstrapped_reward, weight=1)
                        X.append(X_new)
                        y.append(y_new)
                        if actor_critic_method:
                            X_new_c, y_new_c = model.return_design_matrix((states[i], 'none'), bootstrapped_reward, weight=1, critic_model=True)
                            X_critic.append(X_new_c)
                            y_critic.append(y_new_c)

                # Train model and reset design matrix
                model.fit(X, y)
                X, y = [], []

                if actor_critic_method:
                    model.fit(X_critic, y_critic, critic_model=True)
                    X_critic, y_critic = [], []

            # Check game status and break if you have a result (printing only makes sense for gridworld)
            if done:
                if not train and display_state and clipped_reward > 0: print 'Player WINS!'
                if not train and display_state and clipped_reward < 0: print 'Player LOSES!'
                break

        # Record end of the episode statistics
        statistics.record_episodic_statistics(done, episode, move, clipped_reward, total_episodic_reward, episodic_max_q,
                                              epsilon=epsilon, train=train, model_class=model_class, thread_id=thread_id)

        # if train:
        #     if thread_id == 4 and episode % 2000 == 0 and episode != epochs:
        #         print "Saving model and continuing------------------"
        #         model.save_and_continue()

    if not train:
        statistics.calculate_summary_statistics(model)
        return statistics.result

    else:
        print "------ Finishing thread:  " + str(thread_id) + " -------------------------"

def generate_training_samples_with_experience_replay(experience_replay_obs, env, model, X, y, episode_key, gamma, state_tuple):
    experience_replay_obs.store_for_experience_replay(state_tuple, episode_key)

    # Start training only after buffer is full
    if experience_replay_obs.start_training():
        # randomly sample our experience replay memory
        minibatch = experience_replay_obs.return_minibatch()

        # Now for each gameplay experience, update current reward based on the future reward (using action given by the model)
        for idx, index in enumerate(minibatch):
            example = experience_replay_obs.return_minibatch_sample(index, count=idx)
            current_state, action, bootstrapped_reward, new_state, done, episode, move, td_error = example
            X_new, y_new = model.return_design_matrix((current_state, action), bootstrapped_reward, weight=1)
            X.append(X_new)
            y.append(y_new)

    return X, y

def return_bootstrapped_reward(env, model, gamma, new_state, done, actor_critic_method=False):
    if not done and model.if_exists():  # non-terminal state
        if not actor_critic_method:
        # Get value estimate for that best action and update EXISTING reward
            max_q_action, max_q_value = bandits.return_action_based_on_greedy_policy(new_state, model, env.action_space)
        else:
            feature_vector, y = model.return_design_matrix((new_state, 'none'), critic_model=True)
            max_q_value = model.predict(feature_vector, critic_model=True)
    else:
        max_q_value = 0
    return max_q_value


class ExperienceReplay(object):
    def __init__(self, type, batchsize, experience_replay_size, minibatch_method='random'):
        self.type = type
        self.experience_replay_size = experience_replay_size
        self.minibatch_method = minibatch_method
        self.experience_replay = None
        self.batchsize = batchsize
        self.all_indices = range(experience_replay_size)
        # For stratified sampling
        self.positive_sample_fraction = 0.1
        self.positive_batchsize = int(self.batchsize * self.positive_sample_fraction)
        self.negative_batchsize = self.batchsize - self.positive_batchsize
        self.experience_replay_positive = None
        self.experience_replay_negative = None
        self.positive_indices = range(int(experience_replay_size*self.positive_sample_fraction))
        self.negative_indices = range(int(experience_replay_size*(1-self.positive_sample_fraction)))
        self.max_positive_idx = None
        self.initialize()

    def initialize(self):
        if self.type == 'deque':
            self.experience_replay = deque(maxlen=self.experience_replay_size)
            pos_size = int(self.experience_replay_size*self.positive_sample_fraction)
            self.experience_replay_positive = deque(maxlen=pos_size)
            neg_size = self.experience_replay_size - pos_size
            self.experience_replay_negative = deque(maxlen=neg_size)

        # elif self.type == 'dict':
        #     # {(episode, move): state_action_reward_tuple}
        #     self.experience_replay = OrderedDict()

    def store_for_experience_replay(self, state_tuple, episode_move_key=None):
        old_state, best_known_decision, cumu_reward, new_state, done, episode, move, td_error = state_tuple
        if len(old_state)>0 and len(new_state)>0:
            if self.type == 'deque':
                if self.minibatch_method != 'stratified':
                    self.experience_replay.appendleft(state_tuple)
                else:
                    if cumu_reward > 0:
                        self.experience_replay_positive.appendleft(state_tuple)
                    else:
                        self.experience_replay_negative.appendleft(state_tuple)

            # elif self.type == 'dict' and episode_move_key:
            #     if len(self.experience_replay) == self.experience_replay_size:
            #         _ = self.experience_replay.popitem(last=False)
            #     self.experience_replay[episode_move_key] = state_tuple

    def return_minibatch(self):
        if self.minibatch_method == 'random':
            if self.type == 'deque':
                minibatch_indices = random.sample(self.all_indices, self.batchsize)

            # elif self.type == 'dict':
            #     minibatch_indices = random.sample(self.experience_replay.keys(), self.batchsize)

        # Only work with deque type
        elif self.minibatch_method == 'prioritized':
            # Simple prioritization based on magnitude of reward
            total_reward_in_ex_replay = sum(max(abs(st[7]), (1.0 / self.experience_replay_size)) for st in self.experience_replay)
            probs = tuple((max(abs(st[7]), (1.0 / self.experience_replay_size)) * 1.0 / total_reward_in_ex_replay for st in self.experience_replay))
            minibatch_indices = list(np.random.choice(self.all_indices, self.batchsize, probs))

        # Only work with deque type
        elif self.minibatch_method == 'stratified':
            if len(self.experience_replay_positive) >= self.positive_batchsize:
                minibatch_indices_positive = random.sample(self.positive_indices, self.positive_batchsize)
            else:
                minibatch_indices_positive = self.positive_indices
            minibatch_indices_negative = random.sample(self.negative_indices, self.negative_batchsize)
            # First positive indices and then negative indices - keep track of this
            minibatch_indices = minibatch_indices_positive + minibatch_indices_negative
            self.max_positive_idx = len(minibatch_indices_positive)

        return minibatch_indices

    def return_minibatch_sample(self, index, count=None):
        if self.minibatch_method == 'random' or self.minibatch_method == 'prioritized':
            result =  self.experience_replay[index]

        elif self.minibatch_method == 'stratified':
            try:
                if count < self.max_positive_idx:
                    result = self.experience_replay_positive[index]
                else:
                    result =  self.experience_replay_negative[index]
            except Exception as e:
                print e

        return result


    def start_training(self):
        """
        Start training only if experience replay memory is full
        """
        if self.minibatch_method == 'random' or self.minibatch_method == 'prioritized':
            start = False if len(self.experience_replay) < self.experience_replay_size else True

        elif self.minibatch_method == 'stratified':
            start = False if len(self.experience_replay_positive) + len(self.experience_replay_negative) < self.experience_replay_size else True

        return start


class Statistics(object):
    def __init__(self, base_folder_name, test):
        self.total_reward = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.result_file = open(base_folder_name + '/result.data', 'w') if not test else None
        self.result = {}
        self.batch_mse_stat = []

    def record_episodic_statistics(self, done, episode, total_moves, step_reward, total_episodic_reward, episodic_max_q, epsilon, train=True,
                                   model_class=None, thread_id=1):
        """
        For now record statistics only if episode is ended OR max steps are done
        """
        avg_max_q = episodic_max_q*1.0 / total_moves
        res_line = 'Game:{0}, total_steps:{1}, total_reward:{2}, final_reward:{3}, avg_q_value:{4}, epsilon:{5}, thread:{6}'.format(episode,
                                                                                                                total_moves,
                                                                                                                total_episodic_reward,
                                                                                                                round(step_reward, 4),
                                                                                                                round(avg_max_q),
                                                                                                                round(epsilon, 4),
                                                                                                                thread_id)
        if train:
            print res_line
            self.result_file.write(res_line+"\n")

        self.total_reward = total_episodic_reward
        self.total_episodes = episode

        model_type = 'random' if not model_class else model_class
        if model_type not in self.result:
            self.result[model_type] = Counter()

        if done:
            if total_episodic_reward > 0:
                self.result[model_type]['player wins'] += 1
            else:
                self.result[model_type]['player loses'] += 1
        else:
            self.result[model_type]['in process'] += 1

    def calculate_summary_statistics(self, model_class=None):
        model_type = 'random' if not model_class else model_class
        if model_type in self.result:
            self.result[model_type]['avgerage_reward_per_episode'] = round(self.total_reward * 1.0 / max(self.total_episodes, 1), 2)
        self.total_reward = 0
        self.total_episodes = 0
        try:
            self.result_file.close()
        except:
            pass

class ModelManager(BaseManager):
    pass


def train_with_threads(environment_params, rl_params, bandit_params, model_params, epochs, num_of_threads, train=True, display_state=False, use_processes=False):
    start_time = time.time()

    # Initialize statistics and model here and pass it as an argument
    test = not train
    model_params['base_folder_name'] = return_base_path(environment_params['env_name'])
    model_params['actor_critic_model'] = rl_params['memory_structure_params'].get('actor_critic_method', False)
    statistics = Statistics(base_folder_name=model_params['base_folder_name'], test=test)

    if not use_processes:
        model = Model(model_params)
        model.initialize(test)
        actor_learner_threads = [
            threading.Thread(target=play_with_environment, args=(environment_params, model, statistics, rl_params, bandit_params, epochs, thread_id, train, display_state)) for
            thread_id in xrange(1, num_of_threads + 1)]

    else:# Multiprocessing process
        # We will need to register Model class if we want to share model object
        # Statistics need to be registered and shared as well
        # TODO Also infact this is slow.. so probably not worth pursuing
        ModelManager.register('Model', Model)
        manager = ModelManager()
        manager.start()
        model = manager.Model(model_params)
        model.initialize(test)
        actor_learner_threads = [
            multiprocessing.Process(target=play_with_environment, args=(environment_params, model, statistics, rl_params, bandit_params, epochs, thread_id, train, display_state)) for
            thread_id in xrange(1, num_of_threads + 1)]

    for t in actor_learner_threads:
        t.start()
    for t in actor_learner_threads:
        t.join()

    if train: model.finish()
    statistics.calculate_summary_statistics(model.return_model_class())
    print "elapsed time:" + str(int(time.time() - start_time))
    return statistics.result


def return_base_path(name):
    directory = os.path.dirname(os.path.realpath(__file__)) + '/solved_environments/' + name
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def make_environment(environment_params):
    if environment_params['env_name'] == 'gridworld':
        from environments.gridworld import GridWorld
        env = GridWorld(environment_params['grid_size'])
    else:
        env = Environment(environment_params['env_name'], grid_size=environment_params['grid_size'], last_n=environment_params['last_n'], delta_preprocessing=environment_params['delta_preprocessing'])
    return env

def test_trained_model_with_random_play(environment_params, test_games, render=False):
    print "---------- Testing policy:-----------"
    base_folder = return_base_path(environment_params['env_name'])
    model = pickle.load(open(base_folder + '/model_obs.pkl', mode='rb'))
    model.initialize(test=True)
    statistics = Statistics(base_folder_name=base_folder, test=True)

    # First test with trained model
    print "---------- Testing trained VW model -------"
    play_with_environment(environment_params, model, statistics, rl_params={'max_steps': 30}, bandit_params={}, epochs=test_games, train=False, display_state=render)
    # Now with random model

    print "---------- Testing Random model -----------"
    model = None
    play_with_environment(environment_params, model, statistics, rl_params={'max_steps': 30}, bandit_params={}, epochs=test_games, train=False, display_state=False)

    return statistics.result