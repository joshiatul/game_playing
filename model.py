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
import mmh3

# TODO Pass learning rate

"""
All models get implemented here
1) simple lookup
2) vw-python wrapper - works wonderfully well
.. For adding new model update design matrix, fit, predict methods
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

    def initialize(self, test):
        if self.model_class == 'lookup':
            self.model = {}

        elif self.model_class == 'vw_python':
            self.model_path = self.base_folder_name + "/model.vw"
            self.cache_path = self.base_folder_name + "/temp.cache"
            if not test:
                self.model = pyvw.vw(quiet=True, l2=self.params['l2'], loss_function=self.params['loss_function'], passes=1, holdout_off=True,
                                     f=self.model_path,  b=self.params['b'], lrq=self.params['lrq'], l=self.params['l'], k=True)
            else:
                self.model = pyvw.vw("--quiet -i {0}".format(self.model_path))

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
            # cache_key = str(mmh3.hash128(repr(decision_state)))
            cache_key = repr(decision_state)
            if cache_key in self.design_matrix_cache:
                input_str = self.design_matrix_cache[cache_key]
                if reward:
                    #fv.set_label_string(str(reward) + " " + str(weight))
                    fv = str(reward) + " " + str(weight) + input_str
                else:
                    fv = input_str

            else:
                state, decision_taken = decision_state
                # Right now features are simply state X decision interaction + single interaction feature representing state and action
                try:
                    # Features are simply pixel-action interactions
                    all_features = [obs + '-' + str(decision_taken) for obs in state]

                # Hmm design matrix for blackjack is different
                except TypeError:
                    # Not needed anymore
                    all_features = ['-'.join([i, str(j), decision_taken]) for i, j in zip(state._fields, state)]

                # TODO Let's hash state feature (this is just too sparse, so why not)
                tag = '_'.join(all_features)
                tag = str(mmh3.hash128(tag))
                all_features_with_interaction = all_features + [tag]

                input = " ".join(all_features_with_interaction)
                input_str = " |sd " + input + '\n'

                # Do this after cache retrieval
                if reward:
                    output = str(reward) + " " + str(weight)
                    fv = output + input_str
                else:
                    fv = input_str

                #fv = self.model.example(fv)

                # Store in cache
                self.design_matrix_cache[cache_key] = input_str

            return fv, reward

    def fit(self, X, y):
        if self.model_class == 'lookup_table':
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
            for _ in xrange(1):
                # May be shuffling not necessary here
                # random.shuffle(X)
                # res = [fv.learn() for fv in X]
                # No need to invoke vw example object, just use lower level learn function
                res = [self.model.learn(fv) for fv in X]
            self.exists = True
            batch_mse = 0
            # TODO Record loss sum(fv.get_loss()**2 for fv in X) / (len(X)*1.0)
            return batch_mse

    def predict(self, test):
        if self.model_class == 'lookup_table':
            if test not in self.model:
                for d in self.all_possible_decisions:
                    self.model[(test[0], d)] = bandit.DecisionState()
            return self.model[test].value_estimate

        elif self.model_class == 'vw_python':
            # test.learn()  # Little wierd that we have to call learn at all for a prediction
            # res = test.get_simplelabel_prediction()
            res = self.model.predict(test)
            return res

