import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Adaboost:
    def __init__(self, T, config=None, random_state=42):
        self.T = T
        self.random_state = random_state
        # Use the config to set the base classifier, default to DecisionTreeClassifier(max_depth=1) if not specified
        if config is None:
            config = {"base_clf": "DecisionTreeClassifier", "params": {"max_depth": 1}}

        base_clf_type = config.get("base_clf", "DecisionTreeClassifier")
        base_clf_params = config.get("params", {"max_depth": 1})
        base_clf_params["random_state"] = random_state  # Ensure random_state is set

        if base_clf_type == "DecisionTreeClassifier":
            self.weak_clfs = [
                DecisionTreeClassifier(**base_clf_params)
                for _ in range(T)
            ]
        else:
            raise ValueError(f"Unsupported base classifier: {base_clf_type}")

        self.αs = []

    def fit(self, x_train, y_train):

        m = x_train.shape[0]

        W = np.ones(m) / m  # should have shape (m,)

        # loop over the boosting iterations
        for t, weak_clf in enumerate(self.weak_clfs):

            # read the docs of the fit method in sklearn.tree.DecisionTreeClassifier to see how the weights can be passed
            weak_clf.fit(x_train, y_train, sample_weight=W)

            # Compute the indicator function Iₜ for each point. This is a (m,) array of 0s and 1s.
            hₜ = weak_clf.predict(x_train)
            Iₜ = (hₜ != y_train).astype(int)

            # Use the indicator function Iₜ in boolean masking to compute the error
            errₜ = np.sum(W * It)
            errₜ = np.clip(errₜ, 1e-10, 1 - 1e-10)

            # Compute the estimator coefficient αₜ
            αₜ = np.log((1 - errₜ) / errₜ)
            self.αs.append(αₜ)

            # Update the weights using the estimator coefficient αₜ and the indicator function Iₜ
            W = W * np.exp(αₜ * It)

            # Normalize the weights
            W = W / np.sum(W)

        return self

    def predict(self, x_val):
        # Compute a (T, m) array of predictions that maps each estimator to its predictions of x_val weighted by its alpha
        weighted_opinions = np.array([
            α * clf.predict(x_val) for α, clf in zip(self.αs, self.weak_clfs)
        ])
        # Now have T evaluations of x_val each weighted (multiplied) by the corresponding alpha,
        # so as per the formula we only need to take the sign of the sum of the different evaluations
        return np.sign(np.sum(weighted_opinions, axis=0))

    def score(self, x_val, y_val):
        y_pred = self.predict(x_val)
        return np.mean(y_pred == y_val)
