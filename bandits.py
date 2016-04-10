import random

"""
All exploration algorithms live here
We will start with Epsilon-greedy but plan to add softmax and others
"""

class DecisionState(object):
    def __init__(self):
        self.count = 0
        self.value_estimate = 0

class BanditAlgorithm(object):
    def __init__(self, params=0):
        self.decision_states = {}
        self.params = params
        self.policy = {}
        self.decisions = None

    # TODO This may belong outside bandit
    def return_decision_reward_tuples(self, state, model):
        q_value_table = []
        for decision in model.all_possible_decisions:
            if model.exists:
                decision_state = (state, decision)
                feature_vector, y = model.return_design_matrix(decision_state)
                reward = model.predict(feature_vector)
                q_value_table.append((decision, reward))

        return q_value_table

    # TODO This may belong outside bandit
    def return_decision_with_max_reward(self, q_value_table):
        q_value_table.sort(key=lambda tup: tup[1], reverse=True)
        return q_value_table[0]


    def return_action_based_on_greedy_policy(self, state, model):

        if model.exists:
            result=()
            q_value_table = self.return_decision_reward_tuples(state, model)
            # Store policy learned so far
            if q_value_table:
                result = self.return_decision_with_max_reward(q_value_table)
        else:
            result = (random.choice(model.all_possible_decisions), 0)

        return result

    def select_decision_given_state(self, state, model=None, algorithm='random'):

        if algorithm == 'epsilon-greedy':
            if random.random() > self.params:

                result = self.return_action_based_on_greedy_policy(state, model)
                if result:
                    best_known_decision, max_reward = result
                    self.policy[state] = [state[0], state[1], best_known_decision, max_reward]

                else:
                    best_known_decision, max_reward = (random.choice(model.all_possible_decisions), 0)

            else:
                best_known_decision, max_reward = (random.choice(model.all_possible_decisions), 0)

            return best_known_decision, max_reward