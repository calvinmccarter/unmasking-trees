from typing import Union, Optional
from copy import deepcopy
import warnings

import numpy as np
import xgboost as xgb

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from utrees.kdi_quantizer import KDIQuantizer


XGBOOST_DEFAULT_KWARGS = {
    'tree_method' : 'hist',
    'verbosity' : 1,
    'objective' : 'binary:logistic',
}


class Baltobot(BaseEstimator):
    def __init__(
        self,
        depth: int = 4,
        xgboost_kwargs: dict = {},
        strategy: str = 'kdiquantile',
        softmax_temp: float = 1.,
        random_state = None,
    ):
        self.depth = depth
        self.xgboost_kwargs = xgboost_kwargs
        self.strategy = strategy
        self.softmax_temp = softmax_temp
        self.random_state = random_state
        assert depth < 10  # 2^10 models is insane

        self.left_child_ = None
        self.right_child_ = None
        self.quantizer_ = KDIQuantizer(n_bins=2, encode='ordinal', strategy=strategy)
        self.encoder_ = LabelEncoder()
        my_xgboost_kwargs = XGBOOST_DEFAULT_KWARGS.copy()
        my_xgboost_kwargs.update(xgboost_kwargs)
        self.xgber_ = xgb.XGBClassifier(**my_xgboost_kwargs)
        self.constant_val_ = None

        if depth > 0:
            self.left_child_ = Baltobot(
                depth=depth - 1,
                xgboost_kwargs=xgboost_kwargs,
                strategy=strategy,
                softmax_temp=softmax_temp,
                random_state=random_state,
            )
            self.right_child_ = Baltobot(
                depth=depth - 1,
                xgboost_kwargs=xgboost_kwargs,
                strategy=strategy,
                softmax_temp=softmax_temp,
                random_state=random_state,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        assert np.isnan(y).sum() == 0
        if y.size == 0:
            self.constant_val_ = 0.
            return self
        elif np.std(y) == 0:
            self.constant_val_ = np.mean(y)
            return self
        self.constant_val_ = None

        self.quantizer_.fit(y.reshape(-1, 1))
        y_quant = self.quantizer_.transform(y.reshape(-1, 1)).ravel()
        self.encoder_.fit(y_quant)
        y_enc = self.encoder_.transform(y_quant)
        self.xgber_.fit(X, y_enc)

        if self.depth > 0:
            left_ixs = y_enc == 0
            right_ixs = y_enc == 1
            self.left_child_.fit(X[left_ixs, :], y[left_ixs])
            self.right_child_.fit(X[right_ixs, :], y[right_ixs])

        return self

    def sample(
        self,
        X: np.ndarray,
    ):
        n, d = X.shape
        rng = check_random_state(self.random_state)

        if self.constant_val_ is not None:
            return np.full((n,), fill_value=self.constant_val_)

        pred_prob = self.xgber_.predict_proba(X)
        with np.errstate(divide='ignore'):
            annealed_logits = np.log(pred_prob) / self.softmax_temp
        pred_prob = np.exp(annealed_logits) / np.sum(np.exp(annealed_logits), axis=1, keepdims=True)
        pred_enc = np.zeros((n,), dtype=int)
        for i in range(n):
            pred_enc[i] = rng.choice(a=len(self.encoder_.classes_), p=pred_prob[i,:])

        if self.depth == 0:
            pred_quant = self.encoder_.inverse_transform(pred_enc)
            pred_val = self.quantizer_.inverse_transform_sample(pred_quant.reshape(-1, 1))
            return pred_val.ravel()
        else:
            left_ixs = pred_enc == 0
            right_ixs = pred_enc == 1
            pred_val = np.zeros((n,))
            if left_ixs.sum() > 0:
                pred_val[left_ixs] = self.left_child_.sample(X[left_ixs, :])
            if right_ixs.sum() > 0:
                pred_val[right_ixs] = self.right_child_.sample(X[right_ixs, :])
            return pred_val

    def score_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Compute the conditional log-likelihood of y given X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        y : array-like of shape (n_samples,)
            Output variable.

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample. These are normalized to be
            probability densities, after exponentiating, though will not
            necessarily exactly sum to 1 due to numerical error.
        """

        n, d = X.shape
        if self.constant_val_ is not None:
            return np.log(y == self.constant_val_)

        pred_prob = self.xgber_.predict_proba(X)
        y_quant = self.quantizer_.transform(y.reshape(-1, 1)).ravel()
        y_enc = self.encoder_.transform(y_quant)
        if self.depth == 0:
            left_ixs = y_enc == 0
            right_ixs = y_enc == 1
            bin_edges = self.quantizer_.bin_edges_[0]
            left_width = bin_edges[1] - bin_edges[0] + 1e-15
            right_width = bin_edges[2] - bin_edges[1] + 1e-15
            scores = np.zeros((n,))
            scores[left_ixs] = np.log(pred_prob[left_ixs, 0] / left_width)
            scores[right_ixs] = np.log(pred_prob[right_ixs, 1] / right_width)
            return scores
        else:
            left_ixs = y_enc == 0
            right_ixs = y_enc == 1
            scores = np.zeros((n,))
            if left_ixs.sum() > 0:
                left_scores = self.left_child_.score_samples(X[left_ixs, :], y[left_ixs])
                scores[left_ixs] = np.log(pred_prob[left_ixs, 0]) + left_scores
            if right_ixs.sum() > 0:
                right_scores = self.right_child_.score_samples(X[right_ixs, :], y[right_ixs])
                scores[right_ixs] = np.log(pred_prob[right_ixs, 1]) + right_scores
            return scores

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Compute the total log probability density under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : array-like of shape (n_samples,)
            Output variable.

        Returns
        -------
        logprob : float
            Total log-likelihood of y given X.
        """
        logprob = np.sum(self.score_samples(X, y))
        return logprob
