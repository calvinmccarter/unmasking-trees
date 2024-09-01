import numpy as np
import pytest
import time

from sklearn.datasets import (
    load_iris,
    make_moons,
)
from sklearn.neighbors import KernelDensity

from utrees import UnmaskingTrees

CREATE_PLOTS = False

@pytest.mark.parametrize("strategy, depth, duplicate_K, min_score", [
    ("kdiquantile", 3, 10, -2.),
    ("kdiquantile", 3, 10, -2.),
    ("kdiquantile", 4, 10, -2.),
    ("kdiquantile", 4, 50, -2.),
    ("kdiquantile", 5, 50, -2.),
])
def test_moons_generate(strategy, depth, duplicate_K, min_score):
    n_upper = 100
    n_lower = 100
    n_generate = 123
    n = n_upper + n_lower
    data, labels = make_moons(
        (n_upper, n_lower), shuffle=False, noise=0.1, random_state=12345)

    utree = UnmaskingTrees(
        depth=depth,
        strategy=strategy,
        duplicate_K=duplicate_K,
        random_state=12345,
    )
    utree.fit(data)
    newdata = utree.generate(n_generate=n_generate)
    if CREATE_PLOTS:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(newdata[:, 0], newdata[:, 1]);
        plt.savefig(f'test_moons_generate-{strategy}-{depth}-{duplicate_K}.pdf')
    assert newdata.shape == (n_generate, 2)

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
    scores = kde.score_samples(newdata)
    assert scores.mean() >= min_score


@pytest.mark.parametrize("strategy, depth, duplicate_K, cast_float32, min_score, k", [
    ("kdiquantile", 3, 10, True, -1.5, 1),
    ("kdiquantile", 3, 10, True, -1.5, 1),
    ("kdiquantile", 4, 10, True, -1.5, 1),
    ("kdiquantile", 4, 50, True, -1.5, 1),
    ("kdiquantile", 5, 50, True, -1.5, 1),
    ("kdiquantile", 3, 10, False, -1.5, 1),
    ("kdiquantile", 3, 10, False, -1.5, 1),
    ("kdiquantile", 4, 10, False, -1.5, 1),
    ("kdiquantile", 4, 50, False, -1.5, 1),
    ("kdiquantile", 5, 50, False, -1.5, 1),
    ("kdiquantile", 3, 10, False, -1.5, 3),
    ("kdiquantile", 3, 10, False, -1.5, 3),
    ("kdiquantile", 4, 10, False, -1.5, 3),
    ("kdiquantile", 4, 50, False, -1.5, 3),
    ("kdiquantile", 5, 50, False, -1.5, 3),
])
def test_moons_impute(strategy, depth, cast_float32, duplicate_K, min_score, k):
    n_upper = 100
    n_lower = 100
    n = n_upper + n_lower
    data, labels = make_moons(
        (n_upper, n_lower), shuffle=False, noise=0.1, random_state=12345)
    data4impute = data.copy()
    data4impute[:, 1] = np.nan
    X=np.concatenate([data, data4impute], axis=0)

    utree = UnmaskingTrees(
        depth=depth,
        strategy=strategy,
        duplicate_K=duplicate_K,
        cast_float32=cast_float32,
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
        plt.savefig(f'test_moons_impute-{strategy}-{depth}-{duplicate_K}.pdf')

    # Tests providing data to impute
    imputedX = utree.impute(n_impute=k, X=data4impute)
    assert imputedX.shape == (k, data4impute.shape[0], data4impute.shape[1])
    nannystate = ~np.isnan(data4impute)
    for kk in range(k):
        imputedXcur = imputedX[kk, :, :]
        
        np.testing.assert_equal(imputedXcur[nannystate], data4impute[nannystate])
        scores = kde.score_samples(imputedXcur)
        assert scores.mean() >= min_score

def test_moons_score_samples():
    n_upper = 100
    n_lower = 100
    n_generate = 123
    n = n_upper + n_lower
    data, labels = make_moons(
        (n_upper, n_lower), shuffle=False, noise=0.1, random_state=12345)

    utree = UnmaskingTrees(random_state=12345)
    utree.fit(data)
    scores = utree.score_samples(data)
    assert scores.shape == (200,)
    assert np.min(scores) > -0.5

@pytest.mark.parametrize('target_type', [
    'continuous', 'categorical', 'integer',
])
def test_iris(target_type):
    my_data = load_iris()
    X, y = my_data['data'], my_data['target']
    n = X.shape[0]
    Xy = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
    ut_model = UnmaskingTrees(random_state=12345)
    ut_model.fit(Xy, quantize_cols=['continuous']*4 + [target_type])
    Xy_gen_utrees = ut_model.generate(n_generate=n)
    col_names = my_data['feature_names'] + ['target_names']
    petalw = col_names.index('petal width (cm)')
    petall = col_names.index('petal length (cm)')
    assert np.unique(Xy_gen_utrees[:, -1]).size == 3
    assert np.unique(Xy_gen_utrees[:, petall]).size > n - 10
    assert 50 <= np.unique(Xy_gen_utrees[:, petalw]).size < 100
