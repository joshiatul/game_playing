from model import Model
from bandits import BanditAlgorithm
import random
import time
import cPickle as pickle
from collections import Counter, deque


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


def train_reinforcement_strategy_temporal_difference(epochs=1, game_obs='blackjack', model_class='lookup_table', algo='q_learning'):
    """
    Temporal difference learning
    """
    start_time = time.time()

    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    model.initialize()
    epsilon = 0.99
    banditAlgorithm = BanditAlgorithm(params=epsilon)
    memory_storage_limit = 20
    memory_storage = deque(maxlen=memory_storage_limit)
    batchsize = 10
    gamma = 0.1 #### Seems very critical to tune this .. lower the better (0.1 works best for continuous reward)
    steps = 1 # More than 1 steps doesn't work well.. why??
    max_steps = 50
    seen_initial_states = []
    wins = 0
    loss = 0
    new_games = 0
    lamb = 0.1 # For td lambda methods

    for episode in xrange(epochs):

        # Initialize game and parameters for this epoch
        # Store rewards for all steps in an episode
        episodic_memory=[]
        game_obs.reset()
        #TODO check what testing is doing
        # if game_obs.state not in set(seen_initial_states):
        #     print "--------------------------------- new initial game"
        #     new_games += 1
        # a = tuple([tuple(obs) for obs in game_obs.state])
        # seen_initial_states.append(a)
        # model.observed_initial_states = set(seen_initial_states)

        banditAlgorithm.params = epsilon
        move = 0
        print 'Game #: {0}'.format(episode)

        # Start playing the game
        for move in xrange(max_steps):

            # Check game status and breakout if already you have a result
            if game_obs.game_status != 'in process':
                break

            model.buffer += 1

            # Store current game state
            old_state = game_obs.state

            # Figure out best action based on policy
            best_known_decision, known_reward = banditAlgorithm.select_decision_given_state(game_obs.state, game_obs.action_space, model, algorithm='epsilon-greedy')
            # Make move to get to a new state and observe reward (Remember this could be a terminal state)
            new_state, reward, done, info = game_obs.step(best_known_decision)
            episodic_memory.append(reward)
            #new_state = game_obs.state

            # Experience replay storage (deque object maintains a queue, so no extra processing needed)
            # If buffer is full, it gets overwritten due to deque magic
            queue_key = (episode, move)
            episodic_memory.append(reward)
            memory_storage.appendleft({queue_key: (old_state, best_known_decision, reward, new_state)})

            # TODO Why can't we keep on allocating observed rewards to previous steps (using TD-lambda rule except the last step of estimation)

            # If buffer not filled, continue and collect sample to train
            if (len(memory_storage) < memory_storage_limit):
                continue

            # Start training only after buffer is full
            else:

                # randomly sample our experience replay memory
                minibatch = random.sample(memory_storage, batchsize)

                # Now for each gameplay experience, update current reward based on the future reward (using action given by the model)
                for memory_dict in minibatch:

                    memory_lst = memory_dict.values()[0]
                    old_state_er, action_er, reward_er, new_state_er = memory_lst

                    # If game hasn't finished OR if no model then we have to update the reward based on future discounted reward
                    if game_obs.game_status == 'in process' and model.exists:  # non-terminal state
                        # Get value estimate for that best action and update EXISTING reward

                        if algo == 'q_learning':
                            result = banditAlgorithm.return_action_based_on_greedy_policy(new_state_er, model, game_obs.action_space)
                            max_reward = result[1]

                        # Update reward for the current step AND for last n stapes (if n is large, we deploy TD-lambda)
                        # TODO TD-lambda
                        if result:
                            reward_er += gamma * max_reward

                    # Design matrix is based on estimate of reward at state,action step t+1
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
        if game_obs.game_status == 'player wins': wins += 1
        if game_obs.game_status == 'player loses': loss += 1
        if epsilon > 0.1:  # decrement epsilon over time
            epsilon -= (1.0 / epochs)

    model.finish()
    elapsed_time = int(time.time() - start_time)
    print ": took time:" + str(elapsed_time)
    print ": total wins:" + str(wins)
    print ": total losses:" + str(loss)
    print ": unique games:" + str(new_games)

    return banditAlgorithm.policy, model


def test_policy_with_random_play(game_obs, model=None):
    print "---------- Testing policy:-----------"
    banditAlgorithm = BanditAlgorithm(params=0.1)
    game_obs.reset()
    print "Initial state:"
    print game_obs.state

    random_stat = Counter()
    for _ in xrange(100):
        mv = 1
        game_obs.reset()
        while game_obs.game_status == 'in process':
            random_action = random.choice(game_obs.action_space)
            random_reward = game_obs.step(random_action)
            mv += 1
            if mv >= 10 and game_obs.game_status == 'in process':
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
    for _ in xrange(100):
        move = 1
        game_obs.reset()
        # rnd_state = random.choice(list(model.observed_initial_states))
        # game_obs.state = rnd_state
        while game_obs.game_status == 'in process':
            new_qval_table = banditAlgorithm.return_decision_reward_tuples(game_obs.state, model, game_obs.action_space)
            best_action, value_estimate = banditAlgorithm.return_decision_with_max_reward(new_qval_table)
            reward = game_obs.step(best_action)
            move += 1
            if move >= 10 and game_obs.game_status == 'in process':
                model_stat['in process'] += 1
                break

        if game_obs.game_status == 'player wins':
            model_stat['player wins'] += 1
        elif game_obs.game_status == 'player loses':
            model_stat['player loses'] += 1

    return random_stat, model_stat
