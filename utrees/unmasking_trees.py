from typing import Union, Optional
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

from utrees.kdi_quantizer import KDIQuantizer


XGBOOST_DEFAULT_KWARGS = {
    'tree_method' : 'hist',
    'verbosity' : 1,
    'objective' : 'multi:softprob',
}


def top_p_sampling(n_bins, probs, rng, top_p):
    """A modified implementation of nucleus sampling.
    For the class straddling the top_p boundary, the probability mass beyond top_p is discarded.
    But this class does not receive zero probability mass, so it differs
    from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317 .
    This is more mathematically elegant, in my humble opinion.

    Parameters
    ----------
    n_bins : int
        The number of classes (ie vocab size) to sample from.

    probs : np.ndarray with n_bins elements
        The sampling probabilities (not logits) for each bin.

    rng : np.random.RandomState
        An instantiated random number generator.
        
    top_p : float in (0, 1]
        Top-p probability filter.
        
    Returns
    -------
    chosen : np.ndarray of size (1,)
        The sampled integer in [0, n_bins).
    """
    probs = probs.ravel()  # currently assumes only one sample
    sort_indices = np.argsort(probs)[::-1]
    sort_probs = probs[sort_indices]
    cumsum_probs = np.cumsum(sort_probs)
    unnorm_probs = np.diff(np.minimum(cumsum_probs, top_p), prepend=0.)
    unnorm_probs = unnorm_probs[np.argsort(sort_indices)]  # undo the sort
    norm_probs = unnorm_probs / np.sum(unnorm_probs)
    chosen = np.array(rng.choice(n_bins, p=norm_probs))
    return chosen


class UnmaskingTrees(BaseEstimator):
    """Performs generative modeling and multiple imputation on tabular data.
    
    This method generates training data by iteratively masking samples,
    then training per-feature XGBoost models to unmask the data.

    Parameters
    ----------
    n_bins : int > 1
        Number of bins for discretizing continuous features.
        This allows modeling feature distributions with multiple modes.

    duplicate_K : int > 0
        Number of random masking orders per actual sample.
        The training dataset will be of size (n_samples * n_dims * duplicate_K, n_dims).

    top_p : 0 < float <= 1
        Nucleus sampling parameter for inference. Smaller number produces less noisy generations.

    xgboost_kwargs : dict
        Arguments for XGBoost classifier.

    strategy : 'kdiquantile', 'quantile', 'uniform', or 'kmeans'
        The quantization strategy for discretizing continuous features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation.


    Attributes
    ----------
    trees_ : list of XGBClassifier
        Fitted XGBoost models.

    constant_vals_ : list of float or None, with length n_dims
        Gives the values of constant features, or None otherwise.

    quantize_cols : str or list of strs
        Saved user-provided quantize_cols input.

    quantize_cols_ : list of bool, with length n_dims
        Whether to apply (and un-apply) discretization to each feature.

    encoders_ : list of (KDIQuantizer, LabelEncoder) or LabelEncoder
        Preprocessors for each feature.

    xgboost_kwargs_ : 
        Args passed to XGBClassifier.

    X_ : np.ndarray (n_samples, n_dims)
        Input data.

    """
    def __init__(
        self,
        n_bins: int = 20,
        duplicate_K: int = 50,
        top_p: float = 0.9,
        xgboost_kwargs: dict = {},
        strategy: str = 'kdiquantile',
        random_state = None,
    ):
        self.n_bins = n_bins
        self.duplicate_K = duplicate_K
        self.top_p = top_p
        self.xgboost_kwargs_ = XGBOOST_DEFAULT_KWARGS.copy()
        self.xgboost_kwargs_.update(xgboost_kwargs)
        self.strategy = strategy
        self.random_state = random_state

        assert 2 <= n_bins
        assert 1 <= duplicate_K
        assert 0 < top_p <= 1
        assert self.xgboost_kwargs_['objective'] in ('binary:logistic', 'multi:softprob')
        assert strategy in ('kdiquantile', 'quantile', 'uniform', 'kmeans')
    
        self.trees_ = None
        self.constant_vals_ = None
        self.quantize_cols_ = None
        self.encoders_ = None
        self.X_ = None

    def fit(
        self,
        X: np.ndarray,
        quantize_cols: Union[str, list] = 'all',
    ):
        """
        Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims)
            Data to be modeled and imputed (possibly with np.nan).

        quantize_cols : 'all', 'none', or list of strs ('continuous', 'categorical', 'integer')
            Whether to apply (and un-apply) discretization to each feature.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        rng = check_random_state(self.random_state)
        assert isinstance(X, np.ndarray)
        n_samples, n_dims = X.shape
        if isinstance(quantize_cols, list):
            assert len(quantize_cols) == n_dims
            self.quantize_cols_ = []
            for d, elt in enumerate(quantize_cols):
                if elt == 'continuous':
                    self.quantize_cols_.append(True)
                elif elt == 'categorical':
                    self.quantize_cols_.append(False)
                elif elt == 'integer':
                    if len(np.unique(X[:, d])) > self.n_bins:
                        self.quantize_cols_.append(True)
                    else:
                        self.quantize_cols_.append(False)
                else:
                    assert elt in ('continuous', 'categorical', 'integer')
        elif quantize_cols == 'none':
            self.quantize_cols_ = [False] * n_dims
        elif quantize_cols == 'all':
            self.quantize_cols_ = [True] * n_dims
        else:
            raise ValueError(f'unexpected quantize_cols: {quantize_cols}')
        self.quantize_cols = deepcopy(quantize_cols)
        
        # Find features with constant vals, to be unmasked before training and inference
        self.constant_vals_ = []
        for d in range(n_dims):
            col_d = X[~np.isnan(X[:, d]), d]
            if len(np.unique(col_d)) == 1:
                self.constant_vals_.append(np.unique(col_d).item())
            else:
                self.constant_vals_.append(None)

        # Fit encoders
        self.encoders_ = []
        for d in range(n_dims):
            if self.quantize_cols_[d]:
                curq = KDIQuantizer(
                    n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
                curq.fit(X[~np.isnan(X[:, d]), d:d+1])
            else:
                curq = LabelEncoder()
                curq.fit(X[~np.isnan(X[:, d]), d])
            self.encoders_.append(curq)

        # Generate training data
        X_train = []
        Y_train = []
        for dupix in range(self.duplicate_K):
            mask_ixs = np.repeat(np.arange(n_dims)[np.newaxis, :], n_samples, axis=0)
            mask_ixs = np.apply_along_axis(rng.permutation, axis=1, arr=mask_ixs) # n_samples, n_dims
            for n in range(n_samples):
                fuller_X = X[n, :]
                for d in range(n_dims):
                    victim_ix = mask_ixs[n, d]
                    if fuller_X[victim_ix] != np.nan:
                        emptier_X = fuller_X.copy()
                        emptier_X[victim_ix] = np.nan
                        X_train.append(emptier_X.reshape(1, -1))
                        Y_train.append(fuller_X.reshape(1, -1))
                        fuller_X = emptier_X
        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train, axis=0)

        # Replace each KDIQuantizer with (KDIQuantizer, LabelEncoder) tuple,
        # because we need to remove bins which are empty before giving to xgboost.
        for d in range(n_dims):
            if self.quantize_cols_[d]:
                train_ixs = ~np.isnan(Y_train[:, d])
                cur_q = self.encoders_[d]
                curY_train = cur_q.transform(Y_train[train_ixs, d:d+1])
                cur_le = LabelEncoder().fit(curY_train.ravel())
                self.encoders_[d] = (cur_q, cur_le)

        # Unmask constant-value columns before training
        for d in range(n_dims):
            if self.constant_vals_[d] is not None:
                X_train[:, d] = self.constant_vals_[d]

        # Fit trees
        self.trees_ = []
        for d in range(n_dims):
            if self.constant_vals_[d] is not None:
                self.trees_.append(None)
                continue

            train_ixs = ~np.isnan(Y_train[:, d])
            if self.quantize_cols_[d]:
                (cur_q, cur_le) = self.encoders_[d]
                curY_train = cur_le.transform(cur_q.transform(Y_train[train_ixs, d:d+1]).ravel())
            else:
                curY_train = self.encoders_[d].transform(Y_train[train_ixs, d])
            curX_train = X_train[train_ixs, :] 
            my_xgboost_kwargs = dict(**self.xgboost_kwargs_)
            if len(np.unique(curY_train)) == 2:
                my_xgboost_kwargs['objective'] = 'binary:logistic'
            xgber = xgb.XGBClassifier(**my_xgboost_kwargs)
            xgber.fit(curX_train, curY_train)
            self.trees_.append(xgber)

        self.X_ = X.copy()

        return self

    def generate(
        self,
        n_generate: int = 1,
    ):
        """
        Generate new data.

        Parameters
        ----------
        n_generate : int > 0
            Desired number of samples.

        Returns
        -------
        X : np.ndarray of size (n_generate, n_dims)
            Generated data.
        """
        _, n_dims = self.X_.shape
        X = np.full(fill_value=np.nan, shape=(n_generate, n_dims))
        for d in range(n_dims):
            if self.constant_vals_[d] is not None:
                X[:, d] = self.constant_vals_[d]
        genX = self.impute(n_impute=1, X=X)[0, :, :]
        return genX

    def impute(
        self,
        n_impute: int = 1,
        X: np.ndarray = None,
    ):
        """
        Generate new data.

        Parameters
        ----------
        n_impute : int > 0
            Desired number multiple imputations per sample.

        X : np.ndarray of size (n_samples, n_dims) or None
            Data to impute. If None, uses the data passed to fit.

        Returns
        -------
        imputedX : np.ndarray of size (n_impute, n_samples, n_dims)
            Imputed data.
        """
        if X is None:
            X = self.X_.copy()
        (n_samples, n_dims) = X.shape
        rng = check_random_state(self.random_state)

        for d in range(n_dims):
            if self.constant_vals_[d] is not None:
                # Only replace nans with constant vals, because this is impute, ya know
                X[np.isnan(X[:, d]), d] = self.constant_vals_[d]

        imputedX = np.repeat(X[np.newaxis, :, :], repeats=n_impute, axis=0) # (n_impute, n_samples, n_dims)           
        for n in range(n_samples):
            to_unmask = np.where(np.isnan(X[n, :]))[0] # (n_to_unmask,)
            unmask_ixs = np.repeat(to_unmask[np.newaxis, :], n_impute, axis=0)  # (n_impute, n_to_unmask)
            unmask_ixs = np.apply_along_axis(rng.permutation, axis=1, arr=unmask_ixs) # (n_impute, n_to_unmask)
            n_to_unmask = unmask_ixs.shape[1]
            for kix in range(n_impute):
                for dix in range(n_to_unmask):
                    unmask_ix = unmask_ixs[kix, dix]
                    pred_probas = self.trees_[unmask_ix].predict_proba(imputedX[kix,[n], :])
                    if self.quantize_cols_[unmask_ix]:
                        (cur_q, cur_le) = self.encoders_[unmask_ix]
                        pred_quant = top_p_sampling(len(cur_le.classes_), pred_probas, rng, self.top_p)
                        pred_quant = cur_le.inverse_transform(pred_quant.reshape(1,))
                        pred_val = cur_q.inverse_transform_sample(pred_quant.reshape(1, 1))
                    else:
                        cur_quant = self.encoders_[unmask_ix]
                        pred_quant = top_p_sampling(len(cur_quant.classes_), pred_probas, rng, self.top_p)
                        pred_val = cur_quant.inverse_transform(pred_quant.reshape(1,))
                    imputedX[kix, n, unmask_ix] = pred_val.item()
        return imputedX
