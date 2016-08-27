from environments.blackjack import BlackJack
import pandas as pd
import rl_learning as rl

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

# Test simple monte-carlo learning for blackjack
def test_training_monte_carlo_for_blackjack(model_class, epochs):
    blackjack = BlackJack()
    policy, model = train_reinforcement_learning_strategy(num_sims=epochs, game_obs=blackjack, model_class=model_class)

    df = pd.DataFrame(policy).T
    df.columns = ['player_value', 'dealer_value', 'decision', 'score']
    policy_Q_table = df.pivot('player_value', 'dealer_value')['decision']
    print policy_Q_table
    policy_Q_score = df.pivot('player_value', 'dealer_value')['score']
    print policy_Q_score

    # Add ipython notebook 3D ghaph

    # Test policy
    rl.test_policy_with_random_play(blackjack, model)

    return policy, model


# Test TD for blackjack
def test_training_TD_for_blackjack(model_class, epochs=5000):
    blackjack = BlackJack()
    policy, model = train_reinforcement_strategy_temporal_difference(epochs=epochs, game_obs=blackjack, model_class=model_class)
    df = pd.DataFrame(policy).T
    df.columns = ['player_value', 'dealer_value', 'decision', 'score']
    policy_Q_table = df.pivot('player_value', 'dealer_value')['decision']
    print policy_Q_table
    policy_Q_score = df.pivot('player_value', 'dealer_value')['score']
    print policy_Q_score

    # Add ipython notebook 3D ghaph

    # Test policy
    rl.test_policy_with_random_play(blackjack)

    return policy, model


#policy, model = test_training_monte_carlo_for_blackjack(model_class='lookup_table', epochs=5000)
#policy, model = test_training_monte_carlo_for_blackjack(model_class='vw_python', epochs=5000)
policy, model = test_training_TD_for_blackjack(model_class='vw_python', epochs=10000)