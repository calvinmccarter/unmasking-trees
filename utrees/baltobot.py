from typing import Union, Optional
from copy import deepcopy
import warnings

import numpy as np
import xgboost as xgb

from ForestDiffusion import ForestDiffusionModel
from ngboost import NGBRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from utrees.kdi_quantizer import KDIQuantizer

XGBOOST_DEFAULT_KWARGS = {
    "tree_method": "hist",
    "verbosity": 1,
    "objective": "binary:logistic",
}

TABPFN_DEFAULT_KWARGS = {
    "N_ensemble_configurations": 1,
    "seed": 42,
    "batch_size_inference": 1,
    "subsample_features": True,
}


class NanTabPFNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        random_state=None,
        **tabpfn_kwargs,
    ):
        # We do not import this at the top, because Python dependencies are a nightmare.
        from tabpfn import TabPFNClassifier

        self.tabpfn = TabPFNClassifier(**tabpfn_kwargs)
        self.rng = check_random_state(random_state)

    def fit(
        self,
        X,
        y,
    ):
        self.Xtrain = X.copy()
        self.ytrain = y.copy()
        self.n_classes = np.unique(y).shape[0]
        return self

    def predict_proba(
        self,
        X,
    ):
        dtype = self.Xtrain.dtype
        X = X.astype(dtype)
        n_test, d = X.shape
        pred_probas = np.zeros((n_test, self.n_classes), dtype=dtype)

        obsXtest = ~np.isnan(X)
        obsXtrain = ~np.isnan(self.Xtrain)
        obs_patterns = np.unique(obsXtest, axis=0)
        n_patterns = obs_patterns.shape[0]
        for pix in range(n_patterns):
            isobs = obs_patterns[[pix], :]
            test_ixs = (obsXtest == isobs).all(axis=1).nonzero()[0]
            obs_ixs = isobs.ravel().nonzero()[0]
            train_ixs = (obsXtrain | ~isobs).all(axis=1).nonzero()[0]
            while train_ixs.shape[0] == 0:
                # No training example covers the observed features in this test example.
                # This should be rare, but we handle it anyways.
                # We introduce fake missingness into test until training covers it.
                isobs[0, self.rng.choice(obs_ixs)] = False
                obs_ixs = isobs.ravel().nonzero()[0]
                train_ixs = (obsXtrain | ~isobs).all(axis=1).nonzero()[0]
            if isobs.sum() == 0:
                curXtrain = np.zeros((self.Xtrain.shape[0], 1), dtype=dtype)
                curytrain = self.ytrain
                curXtest = np.zeros((X[test_ixs, :].shape[0], 1), dtype=dtype)
            else:
                curXtrain = self.Xtrain[np.ix_(train_ixs, obs_ixs)]
                curytrain = self.ytrain[train_ixs]
                curXtest = X[np.ix_(test_ixs, obs_ixs)]
            if curXtrain.shape[0] > 1024:
                # warnings.warn(f'TabPFN must shrink from {curXtrain.shape[0]} to 1024 samples')
                sixs = self.rng.choice(curXtrain.shape[0], 1024, replace=False)
                curXtrain = curXtrain[sixs, :]
                curytrain = curytrain[sixs]
            if np.unique(curytrain).shape[0] == 1:
                # Removing missingness might make us only see one label
                pred_probas[test_ixs, curytrain[0]] = 1.0
            else:
                self.tabpfn.fit(curXtrain, curytrain)
                cur_pred_prob = self.tabpfn.predict_proba(curXtest)
                pred_probas[test_ixs, :] = cur_pred_prob
        return pred_probas


class Baltobot(BaseEstimator):
    """Performs probabilistic prediction using a BALanced Tree Of BOosted Trees.

    Parameters
    ----------
    depth : int >= 0
        Depth of balanced binary tree for recursively quantizing each feature.
        The total number of quantization bins is 2^depth.

    clf_kwargs : dict
        Arguments for XGBoost (or TabPFN) classifier.

    strategy : 'kdiquantile', 'quantile', 'uniform', 'kmeans'
        The quantization strategy for discretizing continuous features.

    softmax_temp : float > 0
        Softmax temperature for sampling from predicted probabilities.
        As temperature decreases below the default of 1, predictions converge
        to the argmax of each conditional distribution.

    tabpfn : bool, or int >= 0
        Whether to use TabPFN instead of XGBoost classifier.

    leaf : 'uniform' (default), 'flow', 'vp', 'ngboost'
        Method for generating data at leaf nodes.

    leaf_kwargs : dict
        Arguments for leaf-node generation method.

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

    clfer_ : XGBClassifier
        Binary classifier that tell us whether to go left or right bin.

    leaf_ : NGBRegressor, ForestDiffusionModel, or None
        Generator model at leaf node

    left_child_, right_child_ : Baltobot
        Next level in the balanced binary tree.
    """

    def __init__(
        self,
        depth: int = 4,
        clf_kwargs: dict = {},
        strategy: str = "kdiquantile",
        softmax_temp: float = 1.0,
        tabpfn: bool = False,
        leaf: str = "uniform",
        leaf_kwargs: dict = {},
        random_state=None,
    ):

        self.depth = depth
        self.clf_kwargs = clf_kwargs
        self.strategy = strategy
        self.softmax_temp = softmax_temp
        self.tabpfn = tabpfn
        self.leaf = leaf
        self.leaf_kwargs = leaf_kwargs
        self.random_state = check_random_state(random_state)
        assert depth < 10  # 2^10 models is insane
        assert leaf in ("uniform", "flow", "vp", "ngboost")

        self.left_child_ = None
        self.right_child_ = None
        self.quantizer_ = KDIQuantizer(
            n_bins=2, strategy=strategy, random_state=self.random_state
        )
        self.encoder_ = LabelEncoder()
        my_kwargs = (
            XGBOOST_DEFAULT_KWARGS | clf_kwargs | {"random_state": self.random_state}
        )
        self.clfer_ = xgb.XGBClassifier(**my_kwargs)

        if self.tabpfn:
            assert self.tabpfn > 0
            my_kwargs = TABPFN_DEFAULT_KWARGS | clf_kwargs
            self.clfer_ = NanTabPFNClassifier(**my_kwargs)

        self.constant_val_ = None

        if depth > 0:
            self.left_child_ = Baltobot(
                depth=depth - 1,
                clf_kwargs=clf_kwargs,
                strategy=strategy,
                softmax_temp=softmax_temp,
                tabpfn=tabpfn,
                leaf=leaf,
                leaf_kwargs=leaf_kwargs,
                random_state=self.random_state,
            )
            self.right_child_ = Baltobot(
                depth=depth - 1,
                clf_kwargs=clf_kwargs,
                strategy=strategy,
                softmax_temp=softmax_temp,
                tabpfn=tabpfn,
                leaf=leaf,
                leaf_kwargs=leaf_kwargs,
                random_state=self.random_state,
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
            self.constant_val_ = 0.0
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

        self.clfer_.fit(X, y_enc)

        if self.depth > 0:
            left_ixs = y_enc == 0
            right_ixs = y_enc == 1
            self.left_child_.fit(X[left_ixs, :], y[left_ixs])
            self.right_child_.fit(X[right_ixs, :], y[right_ixs])
        else:
            if self.leaf == "uniform":
                pass
            elif self.leaf in ("flow", "vp"):
                self.leaf_ = ForestDiffusionModel(
                    X=y.reshape(-1, 1),
                    X_covs=X,
                    diffusion_type=self.leaf,
                    **self.leaf_kwargs,
                )
            elif self.leaf == "ngboost":
                self.leaf_ = NGBRegressor(verbose=False, **self.leaf_kwargs).fit(X, y)

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

        pred_prob = self.clfer_.predict_proba(X)

        with np.errstate(divide="ignore"):
            annealed_logits = np.log(pred_prob) / self.softmax_temp
        pred_prob = np.exp(annealed_logits) / np.sum(
            np.exp(annealed_logits), axis=1, keepdims=True
        )
        pred_enc = np.zeros((n,), dtype=int)
        for i in range(n):
            pred_enc[i] = rng.choice(a=len(self.encoder_.classes_), p=pred_prob[i, :])

        if self.depth == 0:
            if self.leaf == "uniform":
                pred_quant = self.encoder_.inverse_transform(pred_enc)
                pred_val = self.quantizer_.inverse_transform_sample(
                    pred_quant.reshape(-1, 1)
                )
            elif self.leaf in ("flow", "vp"):
                pred_val = self.leaf_.generate(batch_size=n, X_covs=X)
            elif self.leaf == "ngboost":
                pred_val = self.leaf_.pred_dist(X=X).sample(m=1)
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
        assert self.leaf == "uniform"
        n, d = X.shape
        rng = check_random_state(self.random_state)
        if self.constant_val_ is not None:
            return np.log(y == self.constant_val_)

        if self.tabpfn:
            XX = X.copy()
            XX[np.isnan(XX)] = rng.normal(size=XX.shape)[np.isnan(XX)]
            pred_prob = self.clfer_.predict_proba(XX)
        else:
            pred_prob = self.clfer_.predict_proba(X)

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
                left_scores = self.left_child_.score_samples(
                    X[left_ixs, :], y[left_ixs]
                )
                scores[left_ixs] = np.log(pred_prob[left_ixs, 0]) + left_scores
            if right_ixs.sum() > 0:
                right_scores = self.right_child_.score_samples(
                    X[right_ixs, :], y[right_ixs]
                )
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
