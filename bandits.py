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
            decision_state = (state, decision)
            feature_vector, y = model.return_design_matrix(decision_state)
            reward = model.predict(feature_vector)
            q_value_table.append(reward)

        return q_value_table

    # TODO This may belong outside bandit
    def return_decision_with_max_reward(self, q_value_table):
        max_value = max(q_value_table)
        max_index = q_value_table.index(max_value)
        # q_value_table.sort(key=lambda tup: tup[1], reverse=True)
        return max_value, max_index

    def return_action_based_on_greedy_policy(self, state, model, all_possible_decisions):

        result = ()
        q_value_table = self.return_decision_reward_tuples(state, model, all_possible_decisions)
        # Store policy learned so far
        if q_value_table:
            max_value, max_index = self.return_decision_with_max_reward(q_value_table)
            result = (all_possible_decisions[max_index], max_value)

        return result

    def select_decision_given_state(self, state, all_possible_decisions, model=None, algorithm='random', test=False):

        rnd = random.random()
        if algorithm == 'random' or not getattr(model, 'exists', None) or (
                    algorithm == 'epsilon-greedy' and rnd < self.params and not test) or not state:
            try:
                result = (random.choice(all_possible_decisions), 0)
            except:
                result = (all_possible_decisions.sample(), 0)

        elif test or (algorithm == 'epsilon-greedy' and rnd >= self.params and state):
            result = self.return_action_based_on_greedy_policy(state, model, all_possible_decisions)

            # Probably no need to save policy (for environments other than blackjack)
            # self.policy[state] = [state[0], state[1], best_known_decision, max_reward]

        best_known_decision, max_reward = result
        return best_known_decision, max_reward

    def decrement_epsilon(self, epochs):
        if self.params > 0.1:
            self.params -= (1.0 / epochs)
