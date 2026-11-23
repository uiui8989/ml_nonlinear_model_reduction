"""Unit tests for the shared metrics and data splits. Run with:  python -m pytest -q

The numbers in test_toy_example are the worked example in the vault note
"Quality criteria" (five points on a line, C and D swapped).
"""

import numpy as np
import pytest

from nldr import data, metrics

TOY_X = np.array([[0.0], [1.0], [3.0], [7.0], [12.0]])
TOY_Y = np.array([[0.0], [1.0], [7.0], [3.0], [12.0]])


def blobs(n=300, d=20, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=5, size=(3, d))
    labels = rng.integers(0, 3, n)
    return centres[labels] + rng.normal(size=(n, d)), labels


# --- The worked example -------------------------------------------------------

def test_toy_example():
    rho, r = metrics.rank_matrix(TOY_X), metrics.rank_matrix(TOY_Y)
    np.testing.assert_allclose(metrics.qnx_curve(rho, r)[:3], [2 / 5, 4 / 10, 13 / 15])
    rnx = metrics.rnx_curve(rho, r)
    np.testing.assert_allclose(rnx, [0.2, -0.2, 7 / 15])
    assert metrics.auc_log(rnx) == pytest.approx((0.2 - 0.1 + 7 / 45) / (11 / 6))
    Q = metrics.coranking_matrix(rho, r)
    np.testing.assert_array_equal(Q, [[2, 1, 2, 0], [1, 0, 2, 2], [2, 2, 1, 0], [0, 2, 0, 3]])
    assert (Q.sum(0) == 5).all() and (Q.sum(1) == 5).all()


def test_qnx_equals_coranking_blocks():
    X, _ = blobs(80)
    Y = X[:, :2]
    rho, r = metrics.rank_matrix(X), metrics.rank_matrix(Y)
    Q = metrics.coranking_matrix(rho, r)
    n = X.shape[0]
    blocks = [Q[:K, :K].sum() / (n * K) for K in range(1, n)]
    np.testing.assert_allclose(metrics.qnx_curve(rho, r), blocks)


# --- Reference embeddings ----------------------------------------------------

def test_identity_is_perfect():
    X, _ = blobs(200)
    rho = metrics.rank_matrix(X)
    np.testing.assert_allclose(metrics.rnx_curve(rho, rho), 1.0)
    assert metrics.trustworthiness_continuity(rho, rho, 10) == (1.0, 1.0)


def test_random_embedding_is_near_zero():
    X, _ = blobs(1000)
    Y = np.random.default_rng(1).normal(size=(1000, 2))
    rnx = metrics.rnx_curve(metrics.rank_matrix(X), metrics.rank_matrix(Y))
    assert abs(metrics.auc_log(rnx)) < 0.02
    assert abs(rnx[100:].mean()) < 0.02


def test_trustworthiness_matches_sklearn():
    from sklearn.manifold import trustworthiness

    X, _ = blobs(200)
    Y = X[:, :2]
    T, _ = metrics.trustworthiness_continuity(metrics.rank_matrix(X), metrics.rank_matrix(Y), 7)
    assert T == pytest.approx(trustworthiness(X, Y, n_neighbors=7))


def test_continuity_is_trustworthiness_with_roles_swapped():
    X, _ = blobs(200)
    Y = X[:, :2]
    rho, r = metrics.rank_matrix(X), metrics.rank_matrix(Y)
    T, C = metrics.trustworthiness_continuity(rho, r, 7)
    T2, C2 = metrics.trustworthiness_continuity(r, rho, 7)
    assert C == pytest.approx(T2) and T == pytest.approx(C2)


def test_oos_identity_is_perfect():
    X, _ = blobs(300)
    rnx = metrics.rnx_curve_oos(X[:200], X[:200], X[200:], X[200:])
    np.testing.assert_allclose(rnx, 1.0)


# --- Data splits ----------------------------------------------------------------

def test_splits_are_disjoint_and_reproducible():
    a = data.make_splits(n=100, seeds=[0, 1])
    b = data.make_splits(n=100, seeds=[0, 1])
    for sa, sb in zip(a, b):
        assert len(set(sa.tr) | set(sa.val) | set(sa.te)) == 300
        np.testing.assert_array_equal(sa.tr, sb.tr)
    assert not np.array_equal(a[0].tr, a[1].tr)
