import numpy as np
import math
from sklearn.base import clone
from scipy.stats import mode
from joblib import Parallel, delayed


class Bagging:
    def __init__(self, model, n_estimators=100, max_samples=1.0, random_state=None):
        # Hyperparameters
        self.n_estimators = n_estimators    # the number of estimators (i.e., T)
        self.max_samples = max_samples      # the ratio determining the size of the bootstrap samples D_1, D_2,..,D_T

        # Reproducibility
        self.random_state = random_state

        # Initialize the estimators by cloning the given model
        self.estimators = [clone(model) for _ in range(self.n_estimators)]

    # We will implement this once and then use it T times to generate D_1, D_2,..,D_T in training
    def _get_random_subset(self, x_data, y_data):        
        # Compute sample_size given dataset size and max_samples
        sample_size = int(self.max_samples * len(x_data))

        # Generating `sample_size` indices from 0 to x_data size -1 with replacement
        # Use np.random.choice
        rand_inds = np.random.choice(len(x_data), size=sample_size, replace=True)

        # Use the random_inds to form the sample D_t from D
        x_data_s = x_data[rand_inds]
        y_data_s = y_data[rand_inds]

        return x_data_s, y_data_s

    def fit(self, x_data, y_data):
        # set labels (will need in prediction)
        self.num_labels  = np.unique(y_data)

        for t, estimator in enumerate(self.estimators):
            # Set random seed for _get_random_subset. We have to add t so next iteration its a different D_t.
            np.random.seed(self.random_state + t)  

            # Call _get_random_subset to get D_t
            x_data_s, y_data_s = self._get_random_subset(x_data, y_data)

            # Fit model M_t on the random bootstrap sample D_t
            estimator.fit(x_data_s, y_data_s)

    # Assume y_preds has dims (n_estimators, num_val_points, num_labels)
    # where y_preds[i,j,k] has the probability of the jth point in the validation set being in the kth class assigned by the ith model
    # thinking of it as a tree of matrices may help (as explained before).
    def _get_soft_vote(self, y_preds):
        # Apply the soft voting equation presented in the notebook
        # By averaging all probabilities for the same label for each point over the estimators
        y_pred_mean = np.mean(y_preds, axis=0)

        # What's the shape of y_pred_mean now? -> It is (num_val_points, num_labels)

        # Apply argmax over the probabilities to get the most probable class of each point by soft voting
        y_pred = np.argmax(y_pred_mean, axis=1)
        return y_pred

    def predict(self, x_val):
        num_val_points = x_val.shape[0]
        y_preds = np.empty((self.n_estimators, num_val_points, len(self.num_labels)))
        y_preds[:] = Parallel(n_jobs=-1)(
            delayed(lambda est: est.predict_proba(x_val))(est) for est in self.estimators
        )
        return self._get_soft_vote(y_preds)

    # Implemented as a gift 🎁.
    def score(self, x_val, y_val):
        y_pred = self.predict(x_val)
        return np.mean(y_pred == y_val)
