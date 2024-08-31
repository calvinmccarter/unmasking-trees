from typing import Union, Optional
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

from utrees.baltobot import Baltobot
from utrees.kdi_quantizer import KDIQuantizer


XGBOOST_DEFAULT_KWARGS = {
    'tree_method' : 'hist',
    'verbosity' : 1,
    'objective' : 'multi:softprob',
}


class UnmaskingTrees(BaseEstimator):
    """Performs generative modeling and multiple imputation on tabular data.
    
    This method generates training data by iteratively masking samples,
    then training per-feature XGBoost models to unmask the data.

    Parameters
    ----------
    duplicate_K : int > 0
        Number of random masking orders per actual sample.
        The training dataset will be of size (n_samples * n_dims * duplicate_K, n_dims).

    xgboost_kwargs : dict
        Arguments for XGBoost classifier.

    strategy : 'kdiquantile', 'quantile', 'uniform', 'kmeans'
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
        duplicate_K: int = 50,
        xgboost_kwargs: dict = {},
        depth: int = 4,
        strategy: str = 'kdiquantile',
        softmax_temp: float = 1.,
        random_state = None,
    ):
        self.duplicate_K = duplicate_K
        self.xgboost_kwargs = xgboost_kwargs
        self.xgboost_kwargs_ = XGBOOST_DEFAULT_KWARGS.copy()
        self.xgboost_kwargs_.update(xgboost_kwargs)
        self.depth = depth
        self.strategy = strategy
        self.softmax_temp = softmax_temp
        self.random_state = random_state

        assert 1 <= duplicate_K
        assert self.xgboost_kwargs_['objective'] in ('binary:logistic', 'multi:softprob')
        assert strategy in ('kdiquantile', 'quantile', 'uniform', 'kmeans', 'hierarchical')
        assert 0 < softmax_temp
    
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
                    self.quantize_cols_.append(True)
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
                enc = None
            else:
                enc = LabelEncoder()
                enc.fit(X[~np.isnan(X[:, d]), d])
            self.encoders_.append(enc)

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
            curX_train = X_train[train_ixs, :]
            if self.quantize_cols_[d]:
                curY_train = Y_train[train_ixs, d]
                tobter = Baltobot(
                    depth=self.depth,
                    xgboost_kwargs=self.xgboost_kwargs,
                    strategy=self.strategy,
                    softmax_temp=self.softmax_temp,
                    random_state=self.random_state,
                )
                tobter.fit(curX_train, curY_train)
                self.trees_.append(tobter)
            else:
                curY_train = self.encoders_[d].transform(Y_train[train_ixs, d])
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
                    if self.quantize_cols_[unmask_ix]:
                        pred_val = self.trees_[unmask_ix].sample(imputedX[kix,[n], :])
                    else:
                        pred_probas = self.trees_[unmask_ix].predict_proba(imputedX[kix,[n], :])
                        with np.errstate(divide='ignore'):
                            annealed_logits = np.log(pred_probas) / self.softmax_temp
                        pred_probas = np.exp(annealed_logits) / np.sum(np.exp(annealed_logits), axis=1)
                        cur_enc = self.encoders_[unmask_ix]
                        pred_enc = rng.choice(a=len(cur_enc.classes_), p=pred_probas.ravel())
                        pred_val = cur_enc.inverse_transform(np.array([pred_enc]))
                    imputedX[kix, n, unmask_ix] = pred_val.item()


        return imputedX

    def score_samples(
        self,
        X: np.ndarray,
        n_evals: int = 1,
    ):
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        n_evals : int > 0

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        (n_samples, n_dims) = X.shape
        rng = check_random_state(self.random_state)

        density = np.zeros((n_samples,))
        for n in range(n_samples):
            cur_density = np.zeros((n_evals,))
            for k in range(n_evals):
                eval_order = rng.permutation(n_dims)
                for dix in range(n_dims):
                    eval_ix = eval_order[dix]
                    if not np.isnan(X[n, eval_ix]):
                        evalX = X[[n], :].copy()
                        evalX[:, eval_order[dix:]] = np.nan
                        probas = self.trees_[eval_ix].predict_proba(evalX)[0, :]
                        if self.quantize_cols_[eval_ix]:
                            (cur_quant, cur_le) = self.encoders_[eval_ix]
                            true_bin = cur_quant.transform(X[[n], eval_ix:eval_ix+1]).astype(int).ravel()
                            bin_width = (
                                cur_quant.bin_edges_[0][true_bin+1] - cur_quant.bin_edges_[0][true_bin]
                            ) + 1e-8
                            true_class = cur_le.transform(true_bin).item()
                            cur_density[k] += np.log(probas[true_class] / bin_width)
                        else:
                            cur_quant = self.encoders_[eval_ix]
                            true_class = cur_quant.transform(X[[n], eval_ix:eval_ix+1]).ravel().item()
                            cur_density[k] += np.log(probas[true_class])
            density[n] = np.mean(cur_density)
        return density
