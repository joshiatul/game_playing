import bandits as bandit
from vowpalwabbit import pyvw
import cPickle as pickle
import mmh3


"""
All models get implemented here
1) simple lookup
2) vw-python wrapper - works wonderfully well
.. For adding new model update design matrix, fit, predict methods
"""

class Model(object):
    def __init__(self, params):
        self.model_class = params['class']
        self.actor_model = {}
        self.critic_model = {}
        self.all_possible_decisions = []
        self.base_folder_name = params['base_folder_name']
        self.actor_model_design_matrix_cache = {}
        self.critic_model_design_matrix_cache = {}
        self.exists = False
        self.params = params
        self.critic_model_exists = params.get('actor_critic_model', False)

    def if_exists(self):
        return self.exists

    def return_model_class(self):
        return self.model_class

    def save_and_continue(self):
        self.actor_model.finish()
        self.actor_model = pyvw.vw("--quiet -i {0}".format(self.actor_model_path))

    def finish(self):
        "Let's pickle only if we are running vw"
        if self.model_class == 'vw_python':
            # Want python object for later use
            # self.X = [ex.finish() for ex in self.X]
            self.actor_model.finish()
            self.actor_model = None
            self.actor_model_design_matrix_cache = {}
            if self.critic_model_exists:
                self.critic_model.finish()
                self.critic_model = None
                self.critic_model_design_matrix_cache = {}
            with open(self.base_folder_name + '/model_obs.pkl', mode='wb') as model_file:
                pickle.dump(self, model_file)

    def initialize(self, test):
        if self.model_class == 'lookup':
            self.actor_model = {}

        elif self.model_class == 'vw_python':
            self.actor_model_path = self.base_folder_name + "/model.vw"
            self.critic_model_path = self.base_folder_name + "/model_critic.vw"

            if not test:
                self.actor_model = pyvw.vw(quiet=True, l2=self.params['l2'], loss_function=self.params['loss_function'], passes=1, holdout_off=True,
                                           f=self.actor_model_path, b=self.params['b'], lrq=self.params['lrq'], l=self.params['l'], k=True)

                if self.critic_model_exists:
                    self.critic_model = pyvw.vw(quiet=True, l2=self.params['l2'], loss_function=self.params['loss_function'], passes=1, holdout_off=True,
                                                f=self.critic_model_path, b=self.params['b'], lrq=self.params['lrq'], l=self.params['l'], k=True)
            else:
                self.actor_model = pyvw.vw("--quiet -i {0}".format(self.actor_model_path))
                if self.critic_model_exists:
                    self.critic_model = pyvw.vw("--quiet -i {0}".format(self.critic_model_path))

    def return_design_matrix(self, decision_state, reward=None, weight=1, critic_model=False):
        """
        Design matrix can simply return catesian product of state and decision
        For now all categorical features
        """
        if self.model_class == 'lookup_table':
            return decision_state, reward

        else:
            # cache_key = str(mmh3.hash128(repr(decision_state)))
            # cache_key = repr(decision_state)
            # if not critic_model and cache_key in self.actor_model_design_matrix_cache:
            #     input_str = self.actor_model_design_matrix_cache[cache_key]
            #     # fv.set_label_string(str(reward) + " " + str(weight))
            #     fv = str(reward) + " " + str(weight) + input_str if reward else input_str
            #
            # elif critic_model and cache_key in self.critic_model_design_matrix_cache:
            #     input_str = self.critic_model_design_matrix_cache[cache_key]
            #     fv = str(reward) + " " + str(weight) + input_str if reward else input_str
            #
            # else:
            state, decision_taken = decision_state
            # Right now features are simply state X decision interaction + single interaction feature representing state and action
            # Features are simply pixel-action interactions
            # all_features = [obs + '-' + str(decision_taken) for obs in state] if not critic_model else [obs for obs in state]
            # tag = '_'.join(all_features)
            # tag = "tag_" + str(mmh3.hash64(tag))
            # all_features_with_interaction = all_features + [tag]

            state_namespace = "|state " + " ".join(state) + " " +  "tag_" + str(mmh3.hash64("_".join(state)))
            decision_namespace = "|decision " + "action_" + str(decision_taken)
            # interaction_namespace = " |interaction " + " ".join(all_features_with_interaction)
            input_str = state_namespace + " " +  decision_namespace + '\n'

            # Do this after cache retrieval
            if reward:
                #weight = 10 if abs(reward) > 5 else 1
                output = str(reward) + " " + str(weight)
                fv = output + input_str
            else:
                fv = input_str

                #fv = self.model.example(fv)

                # Store in cache
                # if not critic_model:
                #     self.actor_model_design_matrix_cache[cache_key] = input_str
                # else:
                #     self.critic_model_design_matrix_cache[cache_key] = input_str

            return fv, reward

    def fit(self, X, y, critic_model=False):
        if self.model_class == 'lookup_table':
            for decision_state in X:
                if decision_state not in self.actor_model:
                    for d in self.all_possible_decisions:
                        self.actor_model[(decision_state[0], d)] = bandit.DecisionState()

                self.actor_model[decision_state].count += 1
                # new pred = old pred + (1/count)*(truth - old pred)
                # Incremental or running average
                updated_value = self.actor_model[decision_state].value_estimate + (1.0 / self.actor_model[decision_state].count) * (
                    y - self.actor_model[decision_state].value_estimate)
                self.actor_model[decision_state].value_estimate = updated_value
            self.exists = True

        elif self.model_class == 'vw_python':
            # Let's use vw as good'old sgd solver
            # res = [fv.learn() for fv in X]
            # No need to invoke vw example object, just use lower level learn function
            if not critic_model:
                res = [self.actor_model.learn(fv) for fv in X]
            else:
                res = [self.critic_model.learn(fv) for fv in X]
            self.exists = True
            # TODO Record loss sum(fv.get_loss()**2 for fv in X) / (len(X)*1.0)
            return

    def predict(self, test, critic_model=False):
        if self.model_class == 'lookup_table':
            if test not in self.actor_model:
                for d in self.all_possible_decisions:
                    self.actor_model[(test[0], d)] = bandit.DecisionState()
            return self.actor_model[test].value_estimate

        elif self.model_class == 'vw_python':
            # test.learn()  # Little wierd that we have to call learn at all for a prediction
            # res = test.get_simplelabel_prediction()
            res = self.actor_model.predict(test) if not critic_model else self.critic_model.predict(test)
            return res
