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
    buffer = 1000
    batchsize = 100
    gamma = 0.9
    h=0

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

                    if game_obs.game_status == 'in process' and model.exists: #non-terminal state
                        # Get q values for the new state, and then choose best action (a single step temporal difference q learning)
                        # Get value estimate for that best action and update EXISTING reward
                        if algo == 'q_learning':
                            result = banditAlgorithm.return_action_based_on_greedy_policy(new_state_er, model)
                            max_reward = result[1]
                        elif algo == 'sarsa':
                            result = banditAlgorithm.select_decision_given_state(new_state_er, model,
                                                                                            algorithm='epsilon-greedy')
                            max_reward = game_obs.play(result[0])

                        if result:
                            reward_er = (reward_er + (gamma * max_reward))

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

## ------------------------------ Eligibility traces do not work - buggy ----------------------------------------------

# # TODO Implement eligibility traces - SOMEHOW STILL THIS DOESN"T WORK AS EXPECTED
# def comeplete_n_steps_and_do_eligibility_traces(game_obs, banditAlgorithm, model, gamma, steps=1, algo='Q'):
#
#         history={}
#         old_state = game_obs.state
#         action, reward_estimate = banditAlgorithm.select_decision_given_state(game_obs.state, model,
#                                                                                                     algorithm='epsilon-greedy')
#         reward = game_obs.play(action)
#         history[0] = (old_state, action, reward)
#         moves = 0
#         terminated = False
#
#         # Do n steps based on whichever algo, record sequence of steps, get to the terminal step and get reward
#         for step in xrange(1, steps+1):
#
#             new_state = game_obs.state
#             # Check if in terminal state
#             if game_obs.game_status == 'in process':
#                 # Get Q value table (for q learning)
#                 if algo=='Q':
#                     result = banditAlgorithm.return_action_based_on_greedy_policy(new_state, model)
#                     future_action, future_reward_estimate = result
#
#                 elif algo == 'sarsa':
#                     result = banditAlgorithm.select_decision_given_state(new_state, model,
#                                                                                     algorithm='epsilon-greedy')
#                     future_action, future_reward_estimate = result
#
#                 # Take action, record reward, new state
#                 future_reward = game_obs.play(future_action)
#
#                 # Store current step
#                 history[step] = (new_state, future_action, future_reward)
#             else:
#
#                 terminated = True
#                 break
#
#             moves = step-1
#
#         # Based on the (hopefully) terminal state reward, update rewards for all earlier states
#         # history = {backstep-1: (history[backstep-1][0], history[backstep-1][1], history[backstep-1][2] + gamma**backstep * history[backstep][2])
#         #            for backstep in range(moves, 0, -1)
#         #            }
#         # TODO Backup only when you reach terminal step
#         # TODO DO not train for every single step.. train may be n times during an episode
#         for backstep in range(moves, 0, -1):
#             # Backup reward only if player won or lost
#             if terminated:
#                 updated_reward_for_prev_state = history[backstep-1][2] + gamma**backstep * history[backstep][2]
#                 history[backstep-1] = (history[backstep-1][0], history[backstep-1][1], updated_reward_for_prev_state)
#
#             # We can create design matrix in the same loop maan
#             state, action, reward = history[backstep]
#             X_new, y_new = model.return_design_matrix((state, action), reward)
#             if model.model_class != 'lookup_table':
#
#                 # TODO Experience replay goes here (and also keep design matrix small and speed up training)
#                 if len(model.X) < 5000:
#                     model.X.append(X_new)
#                     model.y.append(y_new)
#
#                 else:
#                     rnd = np.random.randint(0, len(model.X)-1)
#                     model.X[rnd] = X_new
#                     model.y[rnd] = y_new
#                     #model.fit(model.X, model.y)
#
#             else:
#                 model.fit([X_new], y_new)
#
#         # # Generate and store design matrix
#         # for step, action_state_reward in history.iteritems():
#         #     state, action, reward = action_state_reward
#         #     X_new, y_new = model.return_design_matrix((state, action), reward)
#         #
#         #     if model.model_class != 'lookup_table':
#         #         model.X.append(X_new)
#         #         model.y.append(y_new)
#         #     else:
#         #         model.fit([X_new], y_new)
#
#         # May be train only once per episode
#         if model.model_class != 'lookup_table':
#             model.fit(model.X, model.y)
#
#             # TODO Basically instead of cleaning buffer we can just overwrite INSTEAD OF appending above!!
#             # TODO THAT IS MY EXPERIENCE REPLAY
#             #model.clean_buffer()
#
#         # Return final updated (state, action, updated_reward) tuples
#         return model
#
# # TODO Still buggy
# def train_reinforcement_strategy_temporal_difference_eligibility_trace(epochs=1, game_obs='blackjack', model_class='lookup_table',algo='q_learning' ):
#     # Initialize model
#
#     start_time = time.time()
#
#     model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
#     model.initialize()
#     epsilon = 0.98
#     banditAlgorithm = BanditAlgorithm(params=epsilon)
#     gamma = 0.9
#
#     model.all_possible_decisions = game_obs.all_possible_decisions
#
#     for _ in xrange(epochs):
#         # Initialize game
#         game_obs.initiate_game()
#         banditAlgorithm.params = epsilon
#         move = 0
#
#         model = comeplete_n_steps_and_do_eligibility_traces(game_obs, banditAlgorithm, model, gamma, steps=40, algo='Q')
#
#         print("Game #: %s" % (_,))
#
#         # TODO Check for terminal state
#         print game_obs.game_status
#         if epsilon > 0.1:  # decrement epsilon over time
#             epsilon -= (1.0 / epochs)
#
#     model.finish()
#     elapsed_time = int(time.time() - start_time)
#     print ": took time:" + str(elapsed_time)
#
#     return banditAlgorithm.policy, model

# ---------------------------------------------------------------------------------------------------

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