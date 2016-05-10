from environments.gridworld.gridworld import GridWorld
import rl_learning as rl

def test_training_TD_for_gridworld(model_class, epochs, train=True):
    gridworld = GridWorld()
    if train:
        policy, model = rl.train_reinforcement_strategy_temporal_difference(epochs=epochs, game_obs=gridworld, model_class=model_class)
    random_stat, model_stat = rl.test_policy_with_random_play(gridworld)
    return random_stat, model_stat

    # Record MSE for each epoch may be?
    # Record % of wins

random_stat, model_stat = test_training_TD_for_gridworld(model_class='vw_python', epochs=5000, train=True)
print random_stat
print model_stat