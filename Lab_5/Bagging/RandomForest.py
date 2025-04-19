import numpy as np
from typing import Optional
from sklearn.tree import DecisionTreeClassifier
from Bagging import Bagging


class RandomForestClassifier(Bagging):

    def __init__(
        self,
        # Bagging Hyperparameters
        n_estimators: int = 100,
        max_samples: float = 1.0,
        # Tree Hyperparameters
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None,
        min_samples_split: int = 2,
        # Common
        random_state: Optional[int] = None,
        **kwargs
    ):
        # Init new parameters. You will use these in the next two TODOs
        self.n_estimators = n_estimators
        self.max_features = max_features or "sqrt"
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # Create a new DecisionTreeClassifier and pass to it the relevant hyperparameters (from self)
        # Which of the features passed is responsible for column subsampling?

        super().__init__(
            model=DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                **kwargs
            ),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )

    # ✅ Random Forest Implementation is done here. Go back to Ensemble.ipynb for a quick test and some analysis.

    # Ignore this function. It allows setting parameters for debugging or visualization.
    def set_params(self, **params):
        # Define the valid parameters for __init__ with default values
        valid_params = {
            "n_estimators": getattr(self, "n_estimators", 100),
            "max_samples": getattr(self, "max_samples", 1.0),
            "max_depth": getattr(self, "max_depth", None),
            "max_features": getattr(self, "max_features", "sqrt"),
            "min_samples_split": getattr(self, "min_samples_split", 2),
            "random_state": getattr(self, "random_state", None),
        }

        # Update with the new parameters
        valid_params.update(params)

        # Explicitly pass only the valid parameters to __init__, excluding **kwargs
        self.__init__(
         n_estimators=valid_params["n_estimators"],
         max_samples=valid_params["max_samples"],
         max_depth=valid_params["max_depth"],
         max_features=valid_params["max_features"],
         min_samples_split=valid_params["min_samples_split"],
         random_state=valid_params["random_state"],
     )
        return self
