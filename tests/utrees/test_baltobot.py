import numpy as np
import pytest

from sklearn.datasets import make_moons

from utrees import Baltobot


def test_moons():
    n_upper = 100
    n_lower = 100
    n_generate = 123
    n = n_upper + n_lower
    data, labels = make_moons(
        (n_upper, n_lower), shuffle=False, noise=0.1, random_state=12345)
    X = data[:, 0:1]
    y = data[:, 1]
    balto = Baltobot(random_state=12345)
    balto.fit(X, y)
    sampley = balto.sample(X)
    assert sampley.shape == (n, )
    evaly = np.linspace(-3, 3, 1000)
    evalX = np.full((1000, 1), fill_value=0.)
    scores = balto.score_samples(evalX, evaly)
    assert scores.shape == (1000,)
    probs = np.exp(scores)
    assert 0 <= np.min(probs)
    assert 0.99 < np.sum(probs)
