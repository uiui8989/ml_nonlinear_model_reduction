"""Unit tests for the criteria and the methods. Run with:  python -m pytest -q

The numbers in test_toy_example are the worked example in the vault note
"Quality criteria" (five points on a line, C and D swapped).
"""

import numpy as np
import pytest

from nldr import methods, metrics

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


# --- Methods: the equivalences from the notes ---------------------------------

def _match_signs(A, B):
    return A * np.sign(np.sum(A * B, axis=0))


def test_linear_kpca_equals_pca():
    X, _ = blobs(200)
    Yp = methods.fit("pca", X, 3).Y
    Yk = methods.fit("kpca", X, 3, {"kernel": "linear"}).Y
    np.testing.assert_allclose(_match_signs(Yk, Yp), Yp, atol=1e-6)


def test_mds_equals_pca_and_gower_extension():
    X, _ = blobs(200)
    pca, mds = methods.fit("pca", X[:150], 3), methods.fit("mds", X[:150], 3)
    np.testing.assert_allclose(_match_signs(mds.Y, pca.Y), pca.Y, atol=1e-6)
    signs = np.sign(np.sum(mds.Y * pca.Y, axis=0))
    np.testing.assert_allclose(mds.transform(X[150:]) * signs, pca.transform(X[150:]), atol=1e-6)


def test_pca_reconstruction_error_is_discarded_spectrum():
    from sklearn.decomposition import PCA

    X, _ = blobs(200)
    full = PCA().fit(X)
    m = methods.fit("pca", X, 4).info["model"]
    err = np.mean(np.sum((X - m.inverse_transform(m.transform(X))) ** 2, axis=1))
    # scikit-learn uses 1/(n-1) for variances, the notes use 1/n
    n = X.shape[0]
    assert err == pytest.approx(full.explained_variance_[4:].sum() * (n - 1) / n)


def test_le_nystrom_is_close_on_training_points():
    X, _ = blobs(300)
    emb = methods.fit("le", X, 2, {"n_neighbors": 15, "tau_factor": None})
    corr = [np.corrcoef(emb.transform(X)[:, a], emb.Y[:, a])[0, 1] for a in range(2)]
    assert min(corr) > 0.95


@pytest.mark.parametrize("name,params", [
    ("isomap", {"n_neighbors": 10}), ("lle", {"n_neighbors": 10}),
    ("le", {"n_neighbors": 10}), ("tsne", {"perplexity": 20}),
])
def test_methods_run_and_beat_chance(name, params):
    X, labels = blobs(300)
    emb = methods.fit(name, X, 2, params)
    assert emb.Y.shape == (300, 2) and np.isfinite(emb.Y).all()
    res = metrics.evaluate(X, emb.Y, labels, Ks=(5,))
    assert res["J"] > 0.05


def test_umap_runs():
    pytest.importorskip("umap")
    X, _ = blobs(300)
    emb = methods.fit("umap", X, 2, {"n_neighbors": 15, "min_dist": 0.1})
    assert emb.transform(X[:10]).shape == (10, 2)


def test_autoencoder_records_history_and_maps_new_points():
    pytest.importorskip("torch")
    X, _ = blobs(300)
    X = (X - X.min()) / (X.max() - X.min())          # pixels-like range [0, 1]
    emb = methods.fit("ae", X[:200], 2, {"hidden": (32,)}, epochs=5, X_val=X[200:])
    assert len(emb.info["history"]["train"]) == 5 and len(emb.info["history"]["val"]) == 5
    assert emb.transform(X[200:]).shape == (100, 2)


# --- The comparison protocol ---------------------------------------------------

def test_protocol_tracks(tmp_path, monkeypatch):
    from nldr import config, data, protocol

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    X, y = blobs(600)
    split = data.make_splits(n=200, seeds=[0], pool_size=600)[0]
    rho_val, rho_te = metrics.rank_matrix(X[split.val]), metrics.rank_matrix(X[split.te])
    grid = [{"n_neighbors": 5}, {"n_neighbors": 15}]
    b = protocol.track_b("isomap", X, y, split, 2, rho_val, rho_te, grid=grid)
    a = protocol.track_a("isomap", X, y, split, 2, grid=grid)
    assert b["Y"].shape == (200, 2) and 0 < b["J"] <= 1 and len(b["selection"]) == 2
    assert a["Y_te"].shape == (200, 2) and 0 < a["J_oos"] <= 1
    assert protocol.track_a("tsne", X, y, split, 2) is None
    # second call is served from the cache and gives the same result
    b2 = protocol.track_b("isomap", X, y, split, 2, rho_val, rho_te, grid=grid)
    np.testing.assert_array_equal(b["Y"], b2["Y"])
    table = protocol.to_table([a, b])
    assert {"J", "J_oos", "params"} <= set(table.columns)


def test_protocol_skips_failing_configurations(tmp_path, monkeypatch):
    from nldr import config, data, protocol

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    X, y = blobs(600)
    split = data.make_splits(n=200, seeds=[0], pool_size=600)[0]
    rho_val, rho_te = metrics.rank_matrix(X[split.val]), metrics.rank_matrix(X[split.te])
    grid = [{"no_such_parameter": 1}, {"kernel": "rbf", "sigma_factor": 1.0}]
    with pytest.warns(UserWarning, match="failed"):
        b = protocol.track_b("kpca", X, y, split, 2, rho_val, rho_te, grid=grid)
    assert b["params"] == "kernel=rbf, sigma_factor=1.0"
