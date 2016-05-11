from environments.blackjack import BlackJack
import pandas as pd
import rl_learning as rl

# Test simple monte-carlo learning for blackjack
def test_training_monte_carlo_for_blackjack(model_class, epochs):
    blackjack = BlackJack()
    policy, model = rl.train_reinforcement_learning_strategy(num_sims=epochs, game_obs=blackjack, model_class=model_class)

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
    policy, model = rl.train_reinforcement_strategy_temporal_difference(epochs=epochs, game_obs=blackjack, model_class=model_class)
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
policy, model = test_training_TD_for_blackjack(model_class='vw_python', epochs=5000)