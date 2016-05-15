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
    def return_decision_reward_tuples(self, state, model, all_possible_decisions):
        q_value_table = []
        for decision in all_possible_decisions:
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


    def return_action_based_on_greedy_policy(self, state, model, all_possible_decisions):

        if model.exists:
            result=()
            q_value_table = self.return_decision_reward_tuples(state, model, all_possible_decisions)
            # Store policy learned so far
            if q_value_table:
                result = self.return_decision_with_max_reward(q_value_table)
        else:
            try:
                result = (random.choice(all_possible_decisions), 0)
            except:
                result = (all_possible_decisions.sample(), 0)

        return result

    def select_decision_given_state(self, state, all_possible_decisions, model=None, algorithm='random'):

        if algorithm == 'epsilon-greedy':
            if random.random() > self.params:

                result = self.return_action_based_on_greedy_policy(state, model, all_possible_decisions)
                if result:
                    best_known_decision, max_reward = result
                    self.policy[state] = [state[0], state[1], best_known_decision, max_reward]

                else:
                    try:
                        best_known_decision, max_reward = (random.choice(all_possible_decisions), 0)
                    except:
                        best_known_decision, max_reward = (all_possible_decisions.sample(), 0)

            else:
                try:
                    best_known_decision, max_reward = (random.choice(all_possible_decisions), 0)
                except:
                    best_known_decision, max_reward = (all_possible_decisions.sample(), 0)

            return best_known_decision, max_reward