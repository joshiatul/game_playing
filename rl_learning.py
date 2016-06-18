from model import Model
from bandits import BanditAlgorithm
import random
import time
import cPickle as pickle
from collections import Counter, deque
import numpy as np
from collections import OrderedDict


# TODO Test arbitary weighnig scheme


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


class RLAgent(object):
    def __init__(self, experience_replay_size, batchsize, gamma, skip_frames, max_steps, minibatch_method='random',
                 train_model_after_samples=1):
        self.gamma = gamma
        self.max_steps = max_steps
        self.skip_frames = skip_frames
        self.frames = deque(maxlen=skip_frames)
        self.train_model_after_samples = train_model_after_samples
        self.experience_replay_obs = ExperienceReplay(type='deque', batchsize=batchsize, experience_replay_size=experience_replay_size,
                                                      minibatch_method=minibatch_method)
        self.statistics = None
        self.model = None
        self.bandit_algorithm = None
        self.batch_mse_stat = []

    def initialize(self, model_params, bandit_params, test=False):
        """
        Initialize model
        Initialize bandit_algorithm
        :param model_params:
        :param bandit_params:
        :return:
        """
        if not test:
            model = Model(model_params)

        else:
            model = pickle.load(open(model_params['base_folder_name'] + '/model_obs.pkl', mode='rb'))

        model.initialize(test)
        bandit_algorithm = BanditAlgorithm(params=bandit_params)

        self.model = model
        self.bandit_algorithm = bandit_algorithm
        self.statistics = Statistics(base_folder_name=model_params['base_folder_name'], test=test)

        return

    def play_with_environment(self, env, epochs, train=True, display_state=False):
        """
        Simple temporal difference learning
        with experience-replay
        :return:
        """
        start_time = time.time()
        for episode in xrange(epochs):

            if not train and display_state: print "Game #-----: " + str(episode)

            # Initialize game and parameters for this epoch
            observation = env.reset()
            current_state = env.preprocess(observation)
            total_reward = 0
            self.batch_mse_stat = []

            if train: self.bandit_algorithm.decrement_epsilon(epochs)

            # Start playing the game
            for move in xrange(self.max_steps):

                if display_state: env.render()

                # Figure out best action based on policy
                best_known_decision, known_reward = self.bandit_algorithm.select_decision_given_state(current_state, env.action_space, self.model,
                                                                                                 algorithm='epsilon-greedy', test=not train)

                observation, cumu_reward, done, info = env.step(best_known_decision, self.skip_frames)
                new_state = env.preprocess(observation)
                cumu_reward = env.clip_reward(cumu_reward, done)
                total_reward += cumu_reward
                td_error = cumu_reward - known_reward

                if train:
                    self.train_q_function_with_experience_replay(env, episode_key=(episode, move), state_tuple=(current_state, best_known_decision,
                                                                                                                cumu_reward, new_state, done, episode, move, td_error))

                # Record statistics
                self.statistics.record_episodic_statistics(done, self.max_steps, episode, move, cumu_reward, total_reward, batch_mse_stat=self.batch_mse_stat,
                                                           epsilon=self.bandit_algorithm.params, train=train, model=self.model)

                # Check game status and break if you have a result
                if done:
                    if not train and display_state and cumu_reward > 0: print 'Player WINS!'
                    if not train and display_state and cumu_reward < 0: print 'Player LOSES!'
                    break

                current_state = new_state

        if train: self.model.finish()
        self.statistics.calculate_summary_statistics(self.model)
        print "elapsed time:" + str(int(time.time() - start_time))
        return self.statistics.result

    def train_q_function_with_experience_replay(self, env, episode_key, state_tuple):
        self.experience_replay_obs.store_for_experience_replay(state_tuple, episode_key)

        # TODO Why can't we keep on allocating observed rewards to previous steps (using TD-lambda rule except the last step of estimation)
        if not self.experience_replay_obs.start_training():
            return

        # Start training only after buffer is full
        else:
            # randomly sample our experience replay memory
            minibatch = self.experience_replay_obs.return_minibatch()

            # Now for each gameplay experience, update current reward based on the future reward (using action given by the model)
            for idx, index in enumerate(minibatch):
                old_state_er, action_er, reward_er, new_state_er, done_er, episode_er, move_er, td_error_er = self.experience_replay_obs.return_minibatch_sample(index, count=idx)

                # If game hasn't finished OR if no model then we have to update the reward based on future discounted reward
                if not done_er and self.model.exists:  # non-terminal state
                    # Get value estimate for that best action and update EXISTING reward
                    result = self.bandit_algorithm.return_action_based_on_greedy_policy(new_state_er, self.model, env.action_space)
                    max_reward = result[1]
                    old_estimate = reward_er - td_error_er
                    reward_er_n = reward_er + self.gamma * max_reward
                    td_error_er = reward_er_n - old_estimate

                else:
                    reward_er_n = reward_er

                # TODO Update TD error if we want any form of prioritized replay
                # self.experience_replay_obs.experience_replay[index] = (
                # old_state_er, action_er, reward_er, new_state_er, done_er, episode_er, move_er, abs(td_error_er))
                # Design matrix is based on estimate of reward at state,action step t+1
                # TODO Use absolute reward as weight may be (or may be some arbitrary scaling of it)
                weight_er = 1
                X_new, y_new = self.model.return_design_matrix((old_state_er, action_er), reward_er_n, weight_er)
                self.model.X.append(X_new)
                self.model.y.append(y_new)

            # We are retraining in every single epoch, but with some subset of all samples
            if len(self.model.X) > self.train_model_after_samples:
                batch_mse = self.model.fit(self.model.X, self.model.y)
                self.batch_mse_stat.append(batch_mse)
                self.model.clean_buffer()

            return

    def test_q_function(self, env, test_games, render=False):
        print "---------- Testing policy:-----------"

        # First test with trained model
        print "---------- Testing trained VW model -------"
        self.play_with_environment(env, epochs=test_games, train=False, display_state=render)
        # Now with random model

        print "---------- Testing Random model -----------"
        self.model = None
        self.play_with_environment(env, epochs=test_games, train=False, display_state=False)

        return self.statistics.result

    # TODO Not using td lambda implementation for now (i assume if i do this, i can get away with less epochs)
    def append_design_matrix_with_td_lambda(self, episode_er, move_er, reward_er, lamb):
        # Do eligibility traces only if reward is substantial
        if reward_er >= abs(-2) and lamb > 0:
            for backstep in reversed(xrange(move_er)):
                weight_er = (1 - lamb) * (lamb ** (move_er - backstep))
                if weight_er < 0.001: break # Break if weight is too low
                # TODO Implement frequency based trace, for now doing only recency based trail
                key1 = (episode_er, backstep)
                if key1 in self.experience_replay_obs.experience_replay:
                    old_state_erl, action_erl, reward_erl, new_state_erl, is_terminal, episode_erl, move_erl = self.experience_replay_obs.experience_replay[key1]
                    reward_erl += self.gamma * reward_er
                    reward_er = reward_erl

                    X_new, y_new = self.model.return_design_matrix((old_state_erl, action_erl), reward_erl, weight_er)
                    self.model.X.append(X_new)
                    self.model.y.append(y_new)


class ExperienceReplay(object):
    def __init__(self, type, batchsize, experience_replay_size, minibatch_method='random'):
        self.type = type
        self.experience_replay_size = experience_replay_size
        self.minibatch_method = minibatch_method
        self.experience_replay = None
        self.batchsize = batchsize
        self.all_indices = range(experience_replay_size)
        # For stratified sampling
        self.positive_batchsize = int(self.batchsize * 0.1)
        self.negative_batchsize = self.batchsize - self.positive_batchsize
        self.experience_replay_positive = None
        self.experience_replay_negative = None
        self.positive_indices = range(int(experience_replay_size*0.1))
        self.negative_indices = range(int(experience_replay_size*(1-0.1)))
        self.max_positive_idx = None
        self.initialize()

    def initialize(self):
        if self.type == 'deque':
            self.experience_replay = deque(maxlen=self.experience_replay_size)
            pos_size = int(self.experience_replay_size*0.1)
            self.experience_replay_positive = deque(maxlen=pos_size)
            neg_size = self.experience_replay_size - pos_size
            self.experience_replay_negative = deque(maxlen=neg_size)

        # elif self.type == 'dict':
        #     # {(episode, move): state_action_reward_tuple}
        #     self.experience_replay = OrderedDict()

    def store_for_experience_replay(self, state_tuple, episode_move_key=None):
        old_state, best_known_decision, cumu_reward, new_state, done, episode, move, td_error = state_tuple
        if old_state and new_state:
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

    def record_episodic_statistics(self, done, max_steps, episode, total_moves, step_reward, total_episodic_reward, batch_mse_stat, epsilon, train=True,
                                   model=None):
        """
        For now record statistics only if episode is ended OR max steps are done
        """
        if done or (not done and total_moves == (max_steps - 1)):
            if batch_mse_stat:
                avg_batch_mse = sum(batch_mse_stat) * 1.0 / len(batch_mse_stat)
            else:
                avg_batch_mse = 0

            res_line = 'Game:{0}; total_steps:{1}; total_reward:{2}; final_reward:{3}; batches_trained:{4}; epsilon:{5}'.format(episode, total_moves,
                                                                                                                    total_episodic_reward,
                                                                                                                    round(step_reward, 4),
                                                                                                                    len(batch_mse_stat),
                                                                                                                    round(epsilon, 4))
            if train:
                print res_line
                self.result_file.write(res_line+"\n")

            self.total_reward = total_episodic_reward
            self.total_episodes = episode

            model_type = 'random' if not model else model.model_class
            if model_type not in self.result:
                self.result[model_type] = Counter()

            if done:
                if total_episodic_reward > 0:
                    self.result[model_type]['player wins'] += 1
                else:
                    self.result[model_type]['player loses'] += 1
            else:
                self.result[model_type]['in process'] += 1

    def calculate_summary_statistics(self, model):
        model_type = 'random' if not model else model.model_class
        self.result[model_type]['avgerage_reward_per_episode'] = round(self.total_reward * 1.0 / self.total_episodes, 2)
        self.total_reward = 0
        self.total_episodes = 0
        try:
            self.result_file.close()
        except:
            pass
