from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import bandits as bandit
from subprocess import Popen, PIPE, STDOUT
import os
from tempfile import NamedTemporaryFile
import sys
from vowpal_wabbit import pyvw
import random
import cPickle as pickle
import hashlib

# TODO Pass learning rate

"""
All models get implemented here
1) simple lookup
2) scikit-learn SGD regressor(doesn't learn)
4) vw-python wrapper - works wonderfully well
5) For adding new model update design matrix, fit, predict methods
"""

class Model(object):
    def __init__(self, params):
        self.model_class = params['class']
        self.model = {}
        self.feature_constructor = None
        self.all_possible_decisions = []
        self.X = []
        self.y = []
        self.buffer = 0
        self.base_folder_name = params['base_folder_name']
        self.design_matrix_cache = {}
        self.exists = False
        self.params = params

    def finish(self):
        "Let's pickle only if we are running vw"
        if self.model_class == 'vw_python':
            # Want python object for later use
            self.X = [ex.finish() for ex in self.X]
            self.model.finish()
            self.X = None
            self.y = None
            self.model = None
            self.design_matrix_cache = {}
            with open(self.base_folder_name + '/model_obs.pkl', mode='wb') as model_file:
                pickle.dump(self, model_file)

    def initialize(self):
        if self.model_class == 'scikit':
            self.model = SGDRegressor(loss='squared_loss', alpha=0.1, n_iter=10, shuffle=True, eta0=0.0001)
            self.feature_constructor = FeatureHasher(n_features=200, dtype=np.float64, non_negative=False, input_type='dict')

        elif self.model_class == 'lookup':
            self.model = {}

        elif self.model_class == 'vw_python':
            self.model_path = self.base_folder_name + "/model.vw"
            self.cache_path = self.base_folder_name + "/temp.cache"
            self.model = pyvw.vw(quiet=True, l2=self.params['l2'], loss_function=self.params['loss_function'], passes=1, holdout_off=True, cache=self.cache_path,
                                 f=self.model_path,  b=self.params['b'], lrq=self.params['lrq'])

    def remove_vw_files(self):
        if os.path.isfile(self.cache_path): os.remove(self.cache_path)
        if os.path.isfile(self.f1): os.remove(self.f1)
        if os.path.isfile(self.model_path): os.remove(self.model_path)

    def clean_buffer(self):
        self.X = []
        self.y = []
        self.buffer = 0

    def return_design_matrix(self, decision_state, reward=None, weight=1):
        """
        Design matrix can simply return catesian product of state and decision
        For now all categorical features
        """
        if self.model_class == 'lookup_table':
            return decision_state, reward

        else:
            train_test_mode = 'train' if reward else 'test'
            cache_key = (decision_state, train_test_mode)
            if cache_key in self.design_matrix_cache:
                fv, reward = self.design_matrix_cache[cache_key]

            else:
                state, decision_taken = decision_state
                # Right now features are simply state X decision interaction + single interaction feature representing state
                try:
                    # Features are simply pixel-action interactions
                    all_features = [obs + '-' + str(decision_taken) for obs in state]

                # Hmm design matrix for blackjack is different
                except TypeError:
                    # Not needed anymore
                    all_features = ['-'.join([i, str(j), decision_taken]) for i, j in zip(state._fields, state)]

                # TODO Let's hash state feature (this is just too sparse, so why not)
                tag = '_'.join(all_features)
                tag = hashlib.md5(tag).hexdigest()
                all_features_with_interaction = all_features + [tag]

                if self.model_class == 'scikit':
                    tr = {fea_value: 1 for fea_value in all_features_with_interaction}
                    fv = self.feature_constructor.transform([tr]).toarray()
                    fv = fv[0]

                elif self.model_class == 'vw' or self.model_class == 'vw_python':
                    input = " ".join(all_features_with_interaction)
                    if reward:
                        output = str(reward) + " " + str(weight)
                        fv = output + " |sd " + input + '\n'
                    else:
                        fv = " |sd " + input + '\n'

                    if self.model_class == 'vw_python':
                        fv = self.model.example(fv)

                # Store only training examples
                # TODO: pyvw for blackjack is somehow still screwed up for cache
                # TODO Something is messed here NEED TO FIX HOw COME ONLY blackjack fails?
                if 'hit' not in self.all_possible_decisions:
                    self.design_matrix_cache[cache_key] = (fv, reward)

            return fv, reward

    def fit(self, X, y):
        if self.model_class == 'scikit':
            # X, y = self.shuffle_data(X, y)
            self.model.partial_fit(X, y)
            print self.model.score(X, y)
            self.exists = True

        elif self.model_class == 'lookup_table':
            for decision_state in X:
                if decision_state not in self.model:
                    for d in self.all_possible_decisions:
                        self.model[(decision_state[0], d)] = bandit.DecisionState()

                self.model[decision_state].count += 1
                # new pred = old pred + (1/count)*(truth - old pred)
                # Incremental or running average
                updated_value = self.model[decision_state].value_estimate + (1.0 / self.model[decision_state].count) * (
                    y - self.model[decision_state].value_estimate)
                self.model[decision_state].value_estimate = updated_value
            self.exists = True

        elif self.model_class == 'vw_python':
            # Let's use vw as good'old sgd solver
            for _ in xrange(20):
                # May be shuffling not necessary here
                random.shuffle(X)
                res = [fv.learn() for fv in X]
            self.exists = True

    def predict(self, test):
        if self.model_class == 'scikit':
            test = test.reshape(1, -1)  # Reshape for single sample
            return self.model.predict(test)[0]

        elif self.model_class == 'lookup_table':
            if test not in self.model:
                for d in self.all_possible_decisions:
                    self.model[(test[0], d)] = bandit.DecisionState()
            return self.model[test].value_estimate

        elif self.model_class == 'vw_python':
            test.learn()  # Little wierd that we have to call learn at all for a prediction
            res = test.get_simplelabel_prediction()
            return res

    @staticmethod
    def shuffle_data(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
