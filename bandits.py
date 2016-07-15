import random
import numpy as np
import math

"""
All exploration algorithms live here
We will start with Epsilon-greedy but plan to add softmax and others
"""


# ---------------- Helper functions for Epsilon-greedy ------------------------- #
def _return_reward_given_decision_state(decision_state, model, critic_model=False):
    feature_vector, y = model.return_design_matrix(decision_state, critic_model=critic_model)
    reward = model.predict(feature_vector, critic_model=critic_model)
    return reward


def return_action_based_on_greedy_policy(state, model, all_possible_decisions):
    result = ()
    q_value_table = tuple(_return_reward_given_decision_state((state, decision), model) for decision in all_possible_decisions)
    # Store policy learned so far
    if q_value_table:
        max_value = max(q_value_table)
        max_index = q_value_table.index(max_value)
        result = (all_possible_decisions[max_index], max_value)
    return result


def return_action_based_on_softmax_policy(state, model, all_possible_decisions):
    if not model or not model.if_exists():
        action = np.random.choice(all_possible_decisions)
        q_val = 0
    else:
        temperature = 1.0
        q_value_table = tuple(_return_reward_given_decision_state((state, decision), model, critic_model=True) for decision in all_possible_decisions)
        q_value_table_exp = tuple(math.exp(q_val / temperature) for q_val in q_value_table)
        total = sum(q_value_table_exp)
        q_value_table_prop = [q_value / total for q_value in q_value_table_exp]
        action = np.random.choice(all_possible_decisions, p=q_value_table_prop)
        q_val = q_value_table[all_possible_decisions.index(action)]

    return action, q_val


def select_action_with_epsilon_greedy_policy(state, all_possible_decisions, model=None, epsilon=0, test=False):
    rnd = random.random()
    if not model or not model.if_exists() or (
                rnd < epsilon and not test) or len(state) == 0:
        try:
            result = (random.choice(all_possible_decisions), 0)
        except:
            result = (all_possible_decisions.sample(), 0)

    elif test or (rnd >= epsilon and len(state) > 0):
        result = return_action_based_on_greedy_policy(state, model, all_possible_decisions)

    best_known_decision, max_reward = result
    return best_known_decision, max_reward


def decrement_epsilon(epochs, epsilon, anneal_epsilon_timesteps, end_epsilon):
    if epsilon > end_epsilon:
        epsilon -= ((1.0-end_epsilon) / anneal_epsilon_timesteps)
    return epsilon


def sample_end_epsilon():
    return np.random.choice([0.1, 0.01, 0.5], p=[0.4, 0.3, 0.3])


# ---- Methods For lookup table -----
class DecisionState(object):
    def __init__(self):
        self.count = 0
        self.value_estimate = 0
