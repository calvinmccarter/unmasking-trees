from typing import List
from typing import Optional

from jaxtyping import Float
from numpy import ndarray
from sklearn.base import MultiOutputMixin
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real

import numpy as np
import utrees

from .base_model import ProbabilisticModel


class BaltoBotModel(ProbabilisticModel, MultiOutputMixin):
    """
    Wrapping the BaltoBot model as a ProbabilisticModel.
    """

    def __init__(
        self,
        learning_rate: float = 0.3,
        max_leaves: int = 0,
        subsample: float = 1,
        seed: int = 0,
    ):
        super().__init__(seed)
        self.subsample = subsample
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.model = None

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: Optional[List[int]] = None,
    ) -> "ProbabilisticModel":
        self.model = utrees.Baltobot(
            random_state=self.seed,
            xgboost_kwargs={
                "subsample": self.subsample,
                "max_leaves": self.max_leaves,
                "learning_rate": self.learning_rate,
            },
        )
        assert y.shape[1] == 1
        self.model.fit(X, y.ravel())
        return self

    def predict(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        y_samples = self.sample(X, n_samples=100)
        y_samples = y_samples.mean(axis=0)
        return y_samples

    def sample(
        self, X: Float[ndarray, "batch x_dim"], n_samples=100, seed=None
    ) -> Float[ndarray, "n_samples batch y_dim"]:
        batch = X.shape[0]
        y_samples = np.zeros((n_samples, batch, 1))
        for n in range(n_samples):
            y_samples[n, :, 0] = self.model.sample(X)
        return y_samples

    @staticmethod
    def search_space() -> dict:
        return {
            "learning_rate": Real(0.05, 0.5, "log-uniform"),
            "max_leaves": Categorical([0, 25, 50]),
            "subsample": Real(0.3, 1., "log-uniform"),
        }

    def get_extra_stats(self) -> dict:
        return {}
