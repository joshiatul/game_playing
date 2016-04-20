from model import Model
from bandits import BanditAlgorithm
from games.blackjack.blackjack import BlackJack
from games.gridworld.gridworld import GridWorld
import pandas as pd
import random
import time
import cPickle as pickle
import numpy as np
from collections import Counter


# TODO : 2) Implement Q learning and SARSA
# TODO:  3) Implement eligibility traces and see how gridworld works


def learn_Q_function(all_observed_decision_states, reward, model):
    # TODO We need to implement experience replay here instead of
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
    start_time = time.time()
    # Initialize model
    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    banditAlgorithm = BanditAlgorithm(params=0.1)
    model.initialize()

    model.all_possible_decisions = game_obs.all_possible_decisions

    for _ in xrange(num_sims):
        model.buffer += 1

        # Initialize game
        game_obs.initiate_game()
        if game_obs.game_status != 'in process':
            continue

        all_observed_decision_states, reward = game_obs.complete_one_episode(banditAlgorithm, model)
        model = learn_Q_function(all_observed_decision_states, reward, model)

    model.finish()
    elapsed_time = int(time.time() - start_time)
    print ": took time:" + str(elapsed_time)
    return banditAlgorithm.policy, model


def train_reinforcement_strategy_temporal_difference(epochs=1, game_obs='blackjack', model_class='lookup_table',algo='q_learning' ):
    # Initialize model

    start_time = time.time()

    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    model.initialize()
    epsilon = 0.99
    banditAlgorithm = BanditAlgorithm(params=epsilon)
    replay = []
    buffer = 500
    batchsize = 100
    gamma = 0.9
    h=0
    steps = 1

    model.all_possible_decisions = game_obs.all_possible_decisions

    for _ in xrange(epochs):
        # Initialize game
        game_obs.initiate_game()
        banditAlgorithm.params = epsilon
        move = 0

        print("Game #: %s" % (_,))

        # TODO This assumes we have a dumb model when we initialize
        while game_obs.game_status == 'in process':
            move += 1

            # Let's start new game if after 10 moves game doesn't end
            if move > 50:
                break

            model.buffer += 1
            old_state = game_obs.state

            # TODO Finish implement q value update using Bellman equation
            best_known_decision, known_reward = banditAlgorithm.select_decision_given_state(game_obs.state, model,
                                                                                            algorithm='epsilon-greedy')
            # Play or make move to get to a new state and see reward
            reward = game_obs.play(best_known_decision)
            new_state = game_obs.state

            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((old_state, best_known_decision, reward, new_state))

            # We do not train until buffer is full, but after that we train with every single epoch
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (old_state, best_known_decision, reward, new_state)

                #randomly sample our experience replay memory
                # TODO We don't need batchsize for vw, we can just replay the whole memory may be
                minibatch = random.sample(replay, batchsize)

                for memory in minibatch:

                    old_state_er, action_er, reward_er, new_state_er = memory

                    for step in xrange(1, steps+1):
                        # TODO This flow is incorrect as well, for different. We should not worry about game status here
                        if game_obs.game_status == 'in process' and model.exists: #non-terminal state
                            # Get q values for the new state, and then choose best action (a single step temporal difference q learning)
                            # Get value estimate for that best action and update EXISTING reward
                            # TODO: Bug here? I think the returned action needs to be played to observe the reward
                            # TODO Instead i am getting the reward "estimate" based on the model
                            if algo == 'q_learning':
                                result = banditAlgorithm.return_action_based_on_greedy_policy(new_state_er, model)
                                # max_reward = result[1]
                                # TODO BAAM i think this is the correct WAY!!!!!
                                max_reward = game_obs.play(result[0])
                            elif algo == 'sarsa':
                                result = banditAlgorithm.select_decision_given_state(new_state_er, model,
                                                                                                algorithm='epsilon-greedy')
                                max_reward = game_obs.play(result[0])

                            if result:
                                reward_er += (gamma**step) * max_reward

                    X_new, y_new = model.return_design_matrix((old_state_er, action_er), reward_er)

                    if model.model_class != 'lookup_table':
                        model.X.append(X_new)
                        model.y.append(y_new)
                    else:
                        model.fit([X_new], y_new)

                # We are retraining in every single epoch, but with some subset of all samples
                if model.model_class != 'lookup_table':
                    model.fit(model.X, model.y)

                    # TODO Instead of killing entire buffer we can keep a few and kill only the subset
                    model.clean_buffer()

            # TODO Check for terminal state
        print game_obs.game_status + " in " + str(move) + " moves"
        if epsilon > 0.1:  # decrement epsilon over time
            epsilon -= (1.0 / epochs)

    model.finish()
    elapsed_time = int(time.time() - start_time)
    print ": took time:" + str(elapsed_time)

    return banditAlgorithm.policy, model


# TODO Compare with random actions
def test_policy_with_random_play(game_obs, model=None):
    print "---------- Testing policy:-----------"
    banditAlgorithm = BanditAlgorithm(params=0.1)
    game_obs.initiate_game()
    print "Initial state:"
    print game_obs.state

    random_stat = Counter()
    mv = 1
    for _ in xrange(100):
        game_obs.initiate_game()
        while game_obs.game_status == 'in process':
            random_action = random.choice(game_obs.all_possible_decisions)
            random_reward = game_obs.play(random_action)
            mv += 1
            if mv >= 50 and game_obs.game_status == 'in process':
                random_stat['in process'] += 1
                break
        if game_obs.game_status == 'player wins':
            random_stat['player wins'] += 1
        elif game_obs.game_status == 'player loses':
            random_stat['player loses'] += 1


    # Unpickle if model obs not provided
    if not model:
        model = pickle.load(open(game_obs.base_folder_name + '/model_obs.pkl', mode='rb'))
    if model.model_class == 'vw_python':
        from vowpal_wabbit import pyvw
        model.model = pyvw.vw("--quiet -i {0}".format(model.model_path))

    model_stat = Counter()
    move = 1
    for _ in xrange(100):
        game_obs.initiate_game()
        while game_obs.game_status == 'in process':
            new_qval_table = banditAlgorithm.return_decision_reward_tuples(game_obs.state, model)
            best_action, value_estimate = banditAlgorithm.return_decision_with_max_reward(new_qval_table)
            reward = game_obs.play(best_action)
            move += 1
            if move >= 50 and game_obs.game_status == 'in process':
                model_stat['in process'] += 1
                break

        if game_obs.game_status == 'player wins':
            model_stat['player wins'] += 1
        elif game_obs.game_status == 'player loses':
            model_stat['player loses'] += 1

    return random_stat, model_stat