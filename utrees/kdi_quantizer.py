# Adapted from scikit-learn/sklearn/preprocessing/_discretization.py
# with the following authorship and license:

# Author: Henry Lin <hlin117@gmail.com>
#         Tom Dupré la Tour

# License: BSD
import warnings
from numbers import Integral

import numpy as np

from kditransform import KDITransformer
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import (
    check_random_state,
    resample,
)
from sklearn.utils._param_validation import Interval, Options, StrOptions
from sklearn.utils.deprecation import _deprecate_Xt_in_inverse_transform
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    check_array,
    check_is_fitted,
)

class KDIQuantizer(TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "n_bins": [Interval(Integral, 2, None, closed="left"), "array-like"],
        "encode": [StrOptions({"ordinal"})],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans", "kdiquantile"})],
        "dtype": [Options(type, {np.float64, np.float32}), None],
        "subsample": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_bins=5,
        *,
        encode="ordinal",
        strategy="kdiquantile",
        dtype=None,
        subsample=200_000,
        random_state=None,
    ):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        sample_weight : ndarray of shape (n_samples,)
            Contains weight values to be associated with each sample.
            Cannot be used when `strategy` is set to `"uniform"`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, dtype="numeric")

        if self.dtype in (np.float64, np.float32):
            output_dtype = self.dtype
        else:  # self.dtype is None
            output_dtype = X.dtype

        n_samples, n_features = X.shape

        if sample_weight is not None and self.strategy in ("uniform", "kdiquantile"):
            raise ValueError(
                "`sample_weight` was provided but it cannot be "
                "used with 'uniform' or 'kdiquantile'. Got strategy="
                f"{self.strategy!r} instead."
            )

        if self.subsample is not None and n_samples > self.subsample:
            # Take a subsample of `X`
            X = resample(
                X,
                replace=False,
                n_samples=self.subsample,
                random_state=self.random_state,
            )

        n_features = X.shape[1]
        n_bins = self._validate_n_bins(n_features)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        bin_edges = np.zeros(n_features, dtype=object)
        for jj in range(n_features):
            column = X[:, jj]
            col_min, col_max = column.min(), column.max()

            if col_min == col_max:
                warnings.warn(
                    "Feature %d is constant and will be replaced with 0." % jj
                )
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.strategy == "uniform":
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)
            elif self.strategy == "quantile":
                quantiles = np.linspace(0, 100, n_bins[jj] + 1)
                if sample_weight is None:
                    bin_edges[jj] = np.asarray(np.percentile(column, quantiles))
                else:
                    bin_edges[jj] = np.asarray(
                        [
                            _weighted_percentile(column, sample_weight, q)
                            for q in quantiles
                        ],
                        dtype=np.float64,
                    )
            elif self.strategy == "kdiquantile":
                quantiles = np.linspace(0, 1, n_bins[jj] + 1)
                kdier = KDITransformer().fit(column.reshape(-1, 1))               
                bin_edges[jj] = kdier.inverse_transform(quantiles.reshape(-1, 1))
            elif self.strategy == "kmeans":
                from ..cluster import KMeans  # fixes import loops

                # Deterministic initialization with uniform spacing
                uniform_edges = np.linspace(col_min, col_max, n_bins[jj] + 1)
                init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5

                # 1D k-means procedure
                km = KMeans(n_clusters=n_bins[jj], init=init, n_init=1)
                centers = km.fit(
                    column[:, None], sample_weight=sample_weight
                ).cluster_centers_[:, 0]
                # Must sort, centers may be unsorted even with sorted init
                centers.sort()
                bin_edges[jj] = (centers[1:] + centers[:-1]) * 0.5
                bin_edges[jj] = np.r_[col_min, bin_edges[jj], col_max]

            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy in ("quantile", "kmeans", "kdiquantile"):
                mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                bin_edges[jj] = bin_edges[jj][mask]
                if len(bin_edges[jj]) - 1 != n_bins[jj]:
                    warnings.warn(
                        "Bins whose width are too small (i.e., <= "
                        "1e-8) in feature %d are removed. Consider "
                        "decreasing the number of bins." % jj
                    )
                    n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        if "onehot" in self.encode:
            self._encoder = OneHotEncoder(
                categories=[np.arange(i) for i in self.n_bins_],
                sparse_output=self.encode == "onehot",
                dtype=output_dtype,
            )
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KDIQuantizer is fitted
            self._encoder.fit(np.zeros((1, len(self.n_bins_))))

        return self

    def _validate_n_bins(self, n_features):
        """Returns n_bins_, the number of bins per feature."""
        orig_bins = self.n_bins
        if isinstance(orig_bins, Integral):
            return np.full(n_features, orig_bins, dtype=int)

        n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)

        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array of shape (n_features,).")

        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError(
                "{} received an invalid number "
                "of bins at indices {}. Number of bins "
                "must be at least 2, and must be an int.".format(
                    KDIQuantizer.__name__, indices
                )
            )
        return n_bins

    def transform(self, X):
        """
        Discretize the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        Xt : {ndarray, sparse matrix}, dtype={np.float32, np.float64}
            Data in the binned space. Will be a sparse matrix if
            `self.encode='onehot'` and ndarray otherwise.
        """
        check_is_fitted(self)

        # check input and attribute dtypes
        dtype = (np.float64, np.float32) if self.dtype is None else self.dtype
        Xt = self._validate_data(X, copy=True, dtype=dtype, reset=False)

        bin_edges = self.bin_edges_
        for jj in range(Xt.shape[1]):
            Xt[:, jj] = np.searchsorted(bin_edges[jj][1:-1], Xt[:, jj], side="right")

        if self.encode == "ordinal":
            return Xt

        dtype_init = None
        if "onehot" in self.encode:
            dtype_init = self._encoder.dtype
            self._encoder.dtype = Xt.dtype
        try:
            Xt_enc = self._encoder.transform(Xt)
        finally:
            # revert the initial dtype to avoid modifying self.
            self._encoder.dtype = dtype_init
        return Xt_enc

    def inverse_transform(self, X=None, *, Xt=None):
        """
        Transform discretized data back to original feature space.

        Note that this function does not regenerate the original data
        due to discretization rounding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Transformed data in the binned space.

        Xt : array-like of shape (n_samples, n_features)
            Transformed data in the binned space.

            .. deprecated:: 1.5
                `Xt` was deprecated in 1.5 and will be removed in 1.7. Use `X` instead.

        Returns
        -------
        Xinv : ndarray, dtype={np.float32, np.float64}
            Data in the original feature space.
        """
        X = _deprecate_Xt_in_inverse_transform(X, Xt)

        check_is_fitted(self)

        if "onehot" in self.encode:
            X = self._encoder.inverse_transform(X)

        Xinv = check_array(X, copy=True, dtype=(np.float64, np.float32))
        n_features = self.n_bins_.shape[0]
        if Xinv.shape[1] != n_features:
            raise ValueError(
                "Incorrect number of features. Expecting {}, received {}.".format(
                    n_features, Xinv.shape[1]
                )
            )

        for jj in range(n_features):
            bin_edges = self.bin_edges_[jj]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            Xinv[:, jj] = bin_centers[(Xinv[:, jj]).astype(np.int64)]

        return Xinv

    def inverse_transform_sample(self, X=None, random_state=None):
        rng = check_random_state(random_state)
        check_is_fitted(self)

        if "onehot" in self.encode:
            X = self._encoder.inverse_transform(X)

        Xinv = check_array(X, copy=True, dtype=(np.float64, np.float32))
        n_features = self.n_bins_.shape[0]
        if Xinv.shape[1] != n_features:
            raise ValueError(
                "Incorrect number of features. Expecting {}, received {}.".format(
                    n_features, Xinv.shape[1]
                )
            )
        n = X.shape[0]
        for jj in range(n_features):
            jitter = rng.uniform(0., 1., size=n)
            bin_edges = self.bin_edges_[jj]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            bin_lefts = bin_edges[1:][(Xinv[:, jj]).astype(np.int64)]
            bin_rights = bin_edges[:-1][(Xinv[:, jj]).astype(np.int64)]
            Xinv[:, jj] = bin_lefts * jitter + bin_rights * (1 - jitter)

        return Xinv

    def get_feature_names_out(self, input_features=None):
        """Get output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(self, input_features)
        if hasattr(self, "_encoder"):
            return self._encoder.get_feature_names_out(input_features)

        # ordinal encoding
        return input_features
