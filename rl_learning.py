from model import Model
from bandits import BanditAlgorithm
import random
import time
import cPickle as pickle
from collections import Counter, deque
import numpy as np


# TODO Reward clipping (clip reward between [-1,1]


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
            model.initialize()

        else:
            # TODO Instantiate a model based on train or test (Move the following in Model)
            model = pickle.load(open(model_params['base_folder_name'] + '/model_obs.pkl', mode='rb'))
            if model.model_class == 'vw_python':
                from vowpal_wabbit import pyvw
                model.model = pyvw.vw("--quiet -i {0}".format(model.model_path))

        bandit_algorithm = BanditAlgorithm(params=bandit_params)
        self.statistics = Statistics(base_folder_name=model_params['base_folder_name'])

        return model, bandit_algorithm

    def play_with_environment(self, env, model, bandit_algorithm, epochs, train=True, display_state=False):
        """
        Simple temporal difference learning
        with experience-replay
        :return:
        """
        for episode in xrange(epochs):

            # Initialize game and parameters for this epoch
            env.reset()
            done = None
            total_reward = 0
            total_steps = 0
            batch_mse_stat = []

            # Start playing the game
            for move in xrange(self.max_steps):

                if display_state: env.render()

                # Check game status and breakout if you have a result
                if done:
                    bandit_algorithm.decrement_epsilon(epochs)
                    break

                # Store current game state
                old_state = env.state if env.state else None

                # Figure out best action based on policy
                best_known_decision, known_reward = bandit_algorithm.select_decision_given_state(env.state, env.action_space, model,
                                                                                                 algorithm='epsilon-greedy', test=not train)

                new_state, cumu_reward, done, info = env.step(best_known_decision, self.skip_frames)
                total_reward += cumu_reward

                if train:
                    if old_state and new_state:
                        self.experience_replay_obs.store_for_experience_replay((old_state, best_known_decision, cumu_reward, new_state, done))

                    # TODO Why can't we keep on allocating observed rewards to previous steps (using TD-lambda rule except the last step of estimation)
                    if not self.experience_replay_obs.start_training():
                        continue

                    # Start training only after buffer is full
                    else:

                        # randomly sample our experience replay memory
                        minibatch = self.experience_replay_obs.return_minibatch()

                        # Now for each gameplay experience, update current reward based on the future reward (using action given by the model)
                        for memory_lst in minibatch:
                            old_state_er, action_er, reward_er, new_state_er, done_er = memory_lst

                            # If game hasn't finished OR if no model then we have to update the reward based on future discounted reward
                            if not done_er and model.exists:  # non-terminal state
                                # Get value estimate for that best action and update EXISTING reward
                                result = bandit_algorithm.return_action_based_on_greedy_policy(new_state_er, model, env.action_space)
                                max_reward = result[1]

                                # Update reward for the current step AND for last n stapes (if n is large, we deploy TD-lambda)
                                # TODO TD-lambda
                                if result:
                                    reward_er += self.gamma * max_reward

                            # Design matrix is based on estimate of reward at state,action step t+1
                            X_new, y_new = model.return_design_matrix((old_state_er, action_er), reward_er)
                            model.X.append(X_new)
                            model.y.append(y_new)

                        # We are retraining in every single epoch, but with some subset of all samples
                        if len(model.X) > self.train_model_after_samples:
                            batch_mse = model.fit(model.X, model.y)
                            batch_mse_stat.append(batch_mse)
                            model.clean_buffer()

                # Record statistics
                self.statistics.record_episodic_statistics(done, self.max_steps, episode, move, cumu_reward, batch_mse_stat=batch_mse_stat,
                                                           train=train, model=model)

        if train: model.finish()
        self.statistics.calculate_final_statistics(model)
        return self.statistics.result

    def test_q_function(self, env, model, bandit_algorithm, test_games, render=False):
        print "---------- Testing policy:-----------"

        self.play_with_environment(env, model=None, bandit_algorithm=bandit_algorithm, epochs=test_games, train=False, display_state=False)
        #random_stat = self.test_q_function_with_model(env, bandit_algorithm, test_games, model='random', render=False)
        #model_stat = self.test_q_function_with_model(env, bandit_algorithm, test_games, model=model, render=render)
        self.play_with_environment(env, model, bandit_algorithm, epochs=test_games, train=False, display_state=False)

        return self.statistics.result

    # def test_q_function_with_model(self, env, bandit_algorithm, test_games=1, model='random', render=False):
    #     result = Counter()
    #     for episode in xrange(test_games):
    #         env.reset()
    #         done = False
    #         per_episode_result = Counter()
    #         for mv in xrange(1, self.max_steps):
    #             if render: env.render()
    #             if model == 'random':
    #                 action = random.choice(env.action_space)
    #             else:
    #                 action, value_estimate = bandit_algorithm.return_action_based_on_greedy_policy(env.state, model, env.action_space)
    #                 observation, reward, done, info = env.step(action, self.skip_frames)
    #
    #             if done:
    #                 if reward > 0:
    #                     result['player wins'] += 1
    #                 else:
    #                     result['player loses'] += 1
    #                 break
    #
    #         if not done:
    #             result['in process'] += 1
    #
    #     result['average reward'] = sum(per_episode_result.itervalues()) / test_games
    #     return result


class ExperienceReplay(object):
    def __init__(self, type, batchsize, experience_replay_size, minibatch_method='random'):
        self.type = type
        self.experience_replay_size = experience_replay_size
        self.minibatch_method = minibatch_method
        self.experience_replay = None
        self.batchsize = batchsize
        self.initialize()

    def initialize(self):
        if self.type == 'deque':
            self.experience_replay = deque(maxlen=self.experience_replay_size)

        elif self.type == 'dict':
            self.experience_replay = {}

    def store_for_experience_replay(self, state_tuple):
        if self.type == 'deque':
            self.experience_replay.appendleft(state_tuple)

        elif self.type == 'dict':
            pass

    def return_minibatch(self):
        if self.minibatch_method == 'random':
            minibatch = random.sample(self.experience_replay, self.batchsize)

        elif self.minibatch_method == 'prioritized':
            # Simple prioritization based on magnitude of reward
            total_reward_in_ex_replay = sum(max(abs(st[2]), (1.0 / self.experience_replay_size)) for st in self.experience_replay)
            probs = tuple((max(abs(st[2]), (1.0 / self.experience_replay_size)) * 1.0 / total_reward_in_ex_replay for st in self.experience_replay))
            selection = set(np.random.choice(range(self.experience_replay_size), self.batchsize, probs))
            minibatch = [j for i, j in enumerate(self.experience_replay) if i in selection]

        return minibatch

    def start_training(self):
        """
        Start training only if experience replay memory is full
        """
        if (len(self.experience_replay) < self.experience_replay_size):
            return False
        else:
            return True

class Statistics(object):
    def __init__(self, base_folder_name):
        self.total_reward = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.result_file = open(base_folder_name + '/result.data', 'w')
        self.result = {}
        self.batch_mse_stat = []

    def record_episodic_statistics(self, done, max_steps, episode, total_moves, total_episodic_reward, batch_mse_stat, train=True,
                                   model=None):
        """
        For now record statistics only if episode is ended OR max steps are done
        """
        if done or (not done and total_moves == (max_steps - 1)):
            if batch_mse_stat:
                avg_batch_mse = sum(batch_mse_stat) * 1.0 / len(batch_mse_stat)
            else:
                avg_batch_mse = 0

            res_line = 'Game:{0}; total_steps:{1}; total_reward:{2}; avg_batch_mse:{3}; batches_trained:{4}'.format(episode, total_moves,
                                                                                                                    total_episodic_reward,
                                                                                                                    round(avg_batch_mse, 2),
                                                                                                                   len(batch_mse_stat))
            if train:
                print res_line
                self.result_file.write(res_line)

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

    def calculate_final_statistics(self, model):
        model_type = 'random' if not model else model.model_class
        self.result[model_type]['avgerage_reward_per_episode'] = round(self.total_reward * 1.0 / self.total_episodes, 2)
        self.total_reward = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.result_file.close()