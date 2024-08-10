import numpy as np
import pytest
import time

from sklearn.datasets import (
    fetch_openml,
    load_iris,
    make_moons,
)
from sklearn.neighbors import KernelDensity

from utrees import UnmaskingTrees

CREATE_PLOTS = True

@pytest.mark.parametrize("strategy, n_bins, duplicate_K, top_p, min_score", [
    ("kdiquantile", 5, 10, 0.9, -2.),
    ("kdiquantile", 5, 10, 1.0, -2.),
    ("kdiquantile", 10, 10, 0.9, -2.),
    ("kdiquantile", 5, 50, 0.9, -2.),
    ("kdiquantile", 20, 50, 0.9, -1.5),
    ("treeffuser", 5, 10, 0.9, -2.),
])
def test_moons_generate(strategy, n_bins, duplicate_K, top_p, min_score):
    n_upper = 100
    n_lower = 100
    n_generate = 123
    n = n_upper + n_lower
    data, labels = make_moons(
        (n_upper, n_lower), shuffle=False, noise=0.1, random_state=12345)

    utree = UnmaskingTrees(
        strategy=strategy,
        n_bins=n_bins,
        duplicate_K=duplicate_K,
        top_p=top_p,
        random_state=12345,
    )
    utree.fit(data)
    newdata = utree.generate(n_generate=n_generate)
    if CREATE_PLOTS:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(newdata[:, 0], newdata[:, 1]);
        plt.savefig(f'test_moons_generate-{strategy}-{n_bins}-{duplicate_K}-{top_p:.2f}.pdf')
    assert newdata.shape == (n_generate, 2)

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
    scores = kde.score_samples(newdata)
    assert scores.mean() >= min_score


@pytest.mark.parametrize("strategy, n_bins, duplicate_K, top_p, min_score, k", [
    ("kdiquantile", 5, 10, 0.9, -2., 1),
    ("kdiquantile", 5, 10, 1.0, -2., 1),
    ("kdiquantile", 10, 10, 0.9, -2., 1),
    ("kdiquantile", 5, 50, 0.9, -2., 1),
    ("kdiquantile", 20, 50, 0.9, -1.5, 1),
    ("kdiquantile", 5, 10, 0.9, -2., 3),
    ("kdiquantile", 5, 10, 1.0, -2., 3),
    ("kdiquantile", 10, 10, 0.9, -2., 3),
    ("kdiquantile", 5, 50, 0.9, -2., 3),
    ("kdiquantile", 20, 50, 0.9, -1.5, 3),
    ("treeffuser", 5, 10, 0.9, -2., 1),
    ("treeffuser", 5, 10, 0.9, -2., 3),
])
def test_moons_impute(strategy, n_bins, duplicate_K, top_p, min_score, k):
    n_upper = 100
    n_lower = 100
    n = n_upper + n_lower
    data, labels = make_moons(
        (n_upper, n_lower), shuffle=False, noise=0.1, random_state=12345)
    data4impute = data.copy()
    data4impute[:, 1] = np.nan
    X=np.concatenate([data, data4impute], axis=0)

    utree = UnmaskingTrees(
        strategy=strategy,
        n_bins=n_bins,
        duplicate_K=duplicate_K,
        top_p=top_p,
        random_state=12345,
    )
    utree.fit(X)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)

    # Tests using fitted data
    imputedX = utree.impute(n_impute=k)
    assert imputedX.shape == (k, X.shape[0], X.shape[1])
    nannystate = ~np.isnan(X)
    for kk in range(k):
        imputedXcur = imputedX[kk, :, :]
        # Assert observed data is unchanged
        np.testing.assert_equal(imputedXcur[nannystate], X[nannystate])
        # Assert all NaNs removed
        assert np.isnan(imputedXcur).sum() == 0
        scores = kde.score_samples(imputedXcur)
        # Assert imputed data is close to observed data distribution
        assert scores.mean() >= min_score
    if CREATE_PLOTS:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(imputedX[0, :, 0], imputedX[0, :, 1]);
        plt.savefig(f'test_moons_impute-{strategy}-{n_bins}-{duplicate_K}-{top_p:.2f}.pdf')

    # Tests providing data to impute
    imputedX = utree.impute(n_impute=k, X=data4impute)
    assert imputedX.shape == (k, data4impute.shape[0], data4impute.shape[1])
    nannystate = ~np.isnan(data4impute)
    for kk in range(k):
        imputedXcur = imputedX[kk, :, :]
        
        np.testing.assert_equal(imputedXcur[nannystate], data4impute[nannystate])
        scores = kde.score_samples(imputedXcur)
        assert scores.mean() >= min_score
