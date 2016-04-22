from model import Model
from bandits import BanditAlgorithm
from games.blackjack.blackjack import BlackJack
from games.gridworld.gridworld import GridWorld
import pandas as pd
import random
import time
import cPickle as pickle
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

# Test TD for gridworld
def test_training_TD_for_gridworld(model_class, epochs, train=True):
    gridworld = GridWorld()
    if train:
        policy, model = rl.train_reinforcement_strategy_temporal_difference(epochs=epochs, game_obs=gridworld, model_class=model_class)
    random_stat, model_stat = rl.test_policy_with_random_play(gridworld)
    return random_stat, model_stat

    # Record MSE for each epoch may be?
    # Record % of wins


if __name__ == "__main__":
    #policy, model = test_training_monte_carlo_for_blackjack(model_class='lookup_table', epochs=5000)
    #policy, model = test_training_monte_carlo_for_blackjack(model_class='vw_python', epochs=5000)
    #policy, model = test_training_TD_for_blackjack(model_class='vw_python', epochs=5000)
    random_stat, model_stat = test_training_TD_for_gridworld(model_class='vw_python', epochs=20000, train=False)
    print random_stat
    print model_stat
    #test_training_TD_lambda_for_gridworld(model_class='vw_python', epochs=1000, train=False)