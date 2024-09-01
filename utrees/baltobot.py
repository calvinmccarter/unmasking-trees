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

    tabpfn: bool
        (Experimental) Whether to use TabPFN instead of XGBoost classifier.

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
        self.random_state = random_state
        assert depth < 10  # 2^10 models is insane

        self.left_child_ = None
        self.right_child_ = None
        self.quantizer_ = KDIQuantizer(n_bins=2, strategy=strategy, random_state=random_state)
        self.encoder_ = LabelEncoder()
        my_xgboost_kwargs = XGBOOST_DEFAULT_KWARGS.copy()
        my_xgboost_kwargs.update(xgboost_kwargs)
        self.xgber_ = xgb.XGBClassifier(**my_xgboost_kwargs)

        if self.tabpfn:
            warnings.warn('Support for TabPFN is experimental.')
            from tabpfn import TabPFNClassifier
            self.xgber_ = TabPFNClassifier()

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
        """Recursively fits balanced tree of boosted tree classifiers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input variables of train set.

        y : array-like of shape (n_samples,)
            Continuous target variable of train set.

        Returns
        -------
        self
        """

        assert np.isnan(y).sum() == 0
        if y.size == 0:
            self.constant_val_ = 0.
            return self
        self.constant_val_ = None

        if self.tabpfn:
            # Remove all nan rows and columns, in order of highest-missingness
            self.tabpfn_cols_ = np.arange(X.shape[1])
            isnanX = np.isnan(X)
            while isnanX.sum() > 0:
                rownans = isnanX.mean(axis=1)  # (n_samples,)
                colnans = isnanX.mean(axis=0)  # (n_features,)
                if rownans.max() >= colnans.max():
                    X = np.delete(X, np.argmax(rownans), axis=0)
                    y = np.delete(y, np.argmax(rownans), axis=0)
                else:
                    X = np.delete(X, np.argmax(colnans), axis=1)
                    self.tabpfn_cols_ = np.delete(self.tabpfn_cols_, np.argmax(colnans))
                isnanX = np.isnan(X)

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
                rng = check_random_state(self.random_state)
                sixs = rng.choice(X.shape[0], 1024, replace=False)
                self.xgber_.fit(X[sixs, :], y_enc[sixs])
            else:
                self.xgber_.fit(X, y_enc)
        else:
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

        n, d = X.shape
        rng = check_random_state(self.random_state)

        if self.constant_val_ is not None:
            return np.full((n,), fill_value=self.constant_val_)

        if self.tabpfn:
            X = X[:, self.tabpfn_cols_]

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
