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
        self.all_possible_decisions = []
        self.base_folder_name = params['base_folder_name']
        self.exists = False
        self.params = params
        self.running_reward = None
        self.epochs = 0

    def if_exists(self):
        return self.exists

    def return_model_class(self):
        return self.model_class

    def increment_epochs(self):
        self.epochs += 1

    def compute_running_reward(self, episode, thread_id, clipped_reward, reward_sum, epsilon):
        self.running_reward = reward_sum if self.running_reward is None else self.running_reward * 0.99 + reward_sum * 0.01
        print "Episode(thread %d): %d  Total reward: %.4f    Running reward: %.4f   With epsilon: %.4f" % (thread_id, episode, clipped_reward,
                                                                                                     self.running_reward, epsilon)
    def save_and_continue(self, thread_id, event):
        if self.epochs % 1000.0 == 0 and thread_id == 1:
            event.clear()
            print "saving model..."
            print "epochs: " + str(self.epochs)
            self.actor_model.finish()
            self.actor_model = pyvw.vw("--quiet --save_resume -f {0} -i {1}".format(self.actor_model_path, self.actor_model_path))
            event.set()

    def finish(self):
        "Let's pickle only if we are running vw"
        if self.model_class == 'vw_python':
            # Want python object for later use
            # self.X = [ex.finish() for ex in self.X]
            self.actor_model.finish()
            self.actor_model = None
            with open(self.base_folder_name + '/model_obs.pkl', mode='wb') as model_file:
                pickle.dump(self, model_file)

    def initialize(self, test, resume=False):
        if self.model_class == 'lookup':
            self.actor_model = {}

        elif self.model_class == 'vw_python':
            self.actor_model_path = self.base_folder_name + "/model.vw"

            if not test:
                if not resume:
                    self.actor_model = pyvw.vw(quiet=True, l2=self.params['l2'], loss_function=self.params['loss_function'], holdout_off=True,
                                           f=self.actor_model_path, b=self.params['b'], lrq=self.params['lrq'], l=self.params['l'], k=True)
                else:
                    self.actor_model = pyvw.vw("--quiet -f {0} -i {0}".format(self.actor_model_path))

            else:
                self.actor_model = pyvw.vw("--quiet -t -i {0}".format(self.actor_model_path))

    def return_design_matrix(self, decision_state, reward=None, weight=1, critic_model=False):
        """
        Design matrix can simply return catesian product of state and decision
        For now all categorical features
        """
        if self.model_class == 'lookup_table':
            return decision_state, reward

        else:
            state, decision_taken = decision_state
            state_namespace = " |state " + " ".join(state) + " " +  "tag_" + str(mmh3.hash128("_".join(state)))
            decision_namespace = " |decision " + "action_" + str(decision_taken)
            input_str = state_namespace + decision_namespace + '\n'

            # Do this after cache retrieval
            if reward:
                output = str(reward) + " " + str(weight)
                fv = output + input_str
            else:
                fv = input_str

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
            res = self.actor_model.predict(test)
            return res
