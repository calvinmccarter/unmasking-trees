from typing import Union, Optional
from copy import deepcopy
import warnings

import numpy as np
import scipy.stats as spst
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
    """Performs probabilistic prediction using a BALanced Tree Of BOosted Trees.

    Parameters
    ----------
    depth : int >= 0
        Depth of balanced binary tree for recursively quantizing each feature.
        The total number of quantization bins is 2^depth.

    xgboost_kwargs : dict
        Arguments for XGBoost classifier.

    strategy : 'kdiquantile', 'quantile', 'uniform', 'kmeans'
        The quantization strategy for discretizing continuous features.

    softmax_temp : float > 0
        Softmax temperature for sampling from predicted probabilities.
        As temperature decreases below the default of 1, predictions converge
        to the argmax of each conditional distribution.

    tabpfn: bool, or int >= 0
        (Experimental) Whether to use TabPFN instead of XGBoost classifier.
        If int >= 1, sets TabPFN N_ensemble_configurations.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation.


    Attributes
    ----------
    constant_val_ : None or float
        Stores the value of a non-varying target variable, avoiding uniform sampling.
        This allows us to sample from discrete or mixed-type random variables.

    quantizer_ : KDIQuantizer
        Maps continuous value to bin. Almost always 2 bins.

    encoder_ : LabelEncoder
        Maps bin (float) to label (int). Almost always 2 classes.

    xgber_ : XGBClassifier
        Binary classifier that tell us whether to go left or right bin.

    left_child_, right_child_ : Baltobot
        Next level in the balanced binary tree.
    """

    def __init__(
        self,
        depth: int = 4,
        xgboost_kwargs: dict = {},
        strategy: str = 'kdiquantile',
        softmax_temp: float = 1.,
        tabpfn: bool = False,
        random_state = None,
    ):

        self.depth = depth
        self.xgboost_kwargs = xgboost_kwargs
        self.strategy = strategy
        self.softmax_temp = softmax_temp
        self.tabpfn = tabpfn
        self.random_state = check_random_state(random_state)
        assert depth < 10  # 2^10 models is insane

        self.left_child_ = None
        self.right_child_ = None
        self.quantizer_ = KDIQuantizer(n_bins=2, strategy=strategy, random_state=self.random_state)
        self.encoder_ = LabelEncoder()
        my_xgboost_kwargs = XGBOOST_DEFAULT_KWARGS | xgboost_kwargs | {'random_state': self.random_state}
        self.xgber_ = xgb.XGBClassifier(**my_xgboost_kwargs)

        if self.tabpfn:
            assert self.tabpfn > 0
            warnings.warn('Support for TabPFN is experimental.')
            from tabpfn import TabPFNClassifier
            self.xgber_ = TabPFNClassifier(N_ensemble_configurations=self.tabpfn, seed=42)

        self.constant_val_ = None

        if depth > 0:
            self.left_child_ = Baltobot(
                depth=depth - 1,
                xgboost_kwargs=xgboost_kwargs,
                strategy=strategy,
                softmax_temp=softmax_temp,
                tabpfn=tabpfn,
                random_state=self.random_state,
            )
            self.right_child_ = Baltobot(
                depth=depth - 1,
                xgboost_kwargs=xgboost_kwargs,
                strategy=strategy,
                softmax_temp=softmax_temp,
                tabpfn=tabpfn,
                random_state=self.random_state,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bandwidth: float = None,
    ):
        """Recursively fits balanced tree of boosted tree classifiers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input variables of train set.

        y : array-like of shape (n_samples,)
            Continuous target variable of train set.

        bandwidth : float
            Bandwidth for KDE. If None, computed using Silverman's rule of thumb.

        Returns
        -------
        self
        """

        assert np.isnan(y).sum() == 0
        if y.size == 0:
            self.constant_val_ = 0.
            return self
        self.constant_val_ = None
        rng = check_random_state(self.random_state)

        self.quantizer_.fit(y.reshape(-1, 1))
        y_quant = self.quantizer_.transform(y.reshape(-1, 1)).ravel()
        self.encoder_.fit(y_quant)
        if len(self.encoder_.classes_) != 2:
            assert len(self.encoder_.classes_) == 1
            assert np.unique(y).shape[0] == 1
            self.constant_val_ = np.mean(y)
            return self
        y_enc = self.encoder_.transform(y_quant)

        if self.tabpfn:
            if X.shape[0] > 1024:
                warnings.warn(f'TabPFN must shrink from {X.shape[0]} to 1024 samples')
                sixs = rng.choice(X.shape[0], 1024, replace=False)
                XX = X[sixs, :]
                yy_enc = y_enc[sixs]
            else:
                XX = X.copy()
                yy_enc = y_enc
            XX[np.isnan(XX)] = rng.normal(size=XX.shape)[np.isnan(XX)]
            self.xgber_.fit(XX, yy_enc)
        else:
            self.xgber_.fit(X, y_enc)

        if bandwidth is None:
            bandwidth = 0.9 * np.power(y.size, -0.2) * np.minimum(np.std(y), spst.iqr(y)/1.34)

        left_ixs = y_enc == 0
        right_ixs = y_enc == 1
        if self.depth > 0:
            self.left_child_.fit(X[left_ixs, :], y[left_ixs], bandwidth=bandwidth)
            self.right_child_.fit(X[right_ixs, :], y[right_ixs], bandwidth=bandwidth)
        else:
            self.left_child_ = (bandwidth, y[left_ixs].copy())
            self.right_child_ = (bandwidth, y[right_ixs].copy())
        return self

    def sample(
        self,
        X: np.ndarray,
    ):
        """Samples y conditional on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input variables of test set.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Samples from the conditional distribution.
        """

        n, _ = X.shape
        rng = check_random_state(self.random_state)

        if self.constant_val_ is not None:
            return np.full((n,), fill_value=self.constant_val_)

        if self.tabpfn:
            XX = X.copy()
            XX[np.isnan(XX)] = rng.normal(size=XX.shape)[np.isnan(XX)]
            pred_prob = self.xgber_.predict_proba(XX)
        else:
            pred_prob = self.xgber_.predict_proba(X)

        with np.errstate(divide='ignore'):
            annealed_logits = np.log(pred_prob) / self.softmax_temp
        pred_prob = np.exp(annealed_logits) / np.sum(np.exp(annealed_logits), axis=1, keepdims=True)
        pred_enc = np.zeros((n,), dtype=int)
        for i in range(n):
            pred_enc[i] = rng.choice(a=len(self.encoder_.classes_), p=pred_prob[i,:])

        left_ixs = pred_enc == 0
        right_ixs = pred_enc == 1
        pred_val = np.zeros((n,))
        if self.depth == 0:
            (left_bw, left_y) = self.left_child_
            (right_bw, right_y) = self.right_child_
            left_n, right_n = left_ixs.sum(), right_ixs.sum()
            pred_val[left_ixs] = rng.choice(left_y, size=left_n) + rng.normal(scale=left_bw, size=left_n)
            pred_val[right_ixs] = rng.choice(right_y, size=right_n) + rng.normal(scale=right_bw, size=right_n)
        else:
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
        rng = check_random_state(self.random_state)
        if self.constant_val_ is not None:
            return np.log(y == self.constant_val_)

        if self.tabpfn:
            XX = X.copy()
            XX[np.isnan(XX)] = rng.normal(size=XX.shape)[np.isnan(XX)]
            pred_prob = self.xgber_.predict_proba(XX)
        else:
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
