"""The nine methods behind one interface.

    emb = fit("isomap", X, q=2, params={"n_neighbors": 10}, seed=0)
    emb.Y              # (n, q) training embedding
    emb.transform(Xn)  # (m, q) embedding of new points, or None if unavailable
    emb.info           # method-specific diagnostics (eigenvalues, graph components, ...)

Samples are ROWS. Method notes in the vault explain every choice made here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist

METHODS = ("pca", "kpca", "mds", "isomap", "lle", "le", "tsne", "umap", "ae")


@dataclass
class Embedding:
    Y: np.ndarray
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def median_distance(X):
    """Median pairwise Euclidean distance (scale for Gaussian bandwidths)."""
    return float(np.median(pdist(X)))


def knn_graph_components(X, n_neighbors):
    """Number of connected components of the symmetrised k-NN graph."""
    from sklearn.neighbors import kneighbors_graph

    G = kneighbors_graph(X, n_neighbors, mode="connectivity", include_self=False)
    return int(connected_components(G.maximum(G.T), directed=False)[0])


# ---------------------------------------------------------------------------
# Projective methods
# ---------------------------------------------------------------------------

def _pca(X, q, seed):
    from sklearn.decomposition import PCA

    m = PCA(n_components=q, svd_solver="full").fit(X)
    return Embedding(m.transform(X), m.transform,
                     {"explained_variance": m.explained_variance_,
                      "explained_variance_ratio": m.explained_variance_ratio_,
                      "model": m})


def _kpca(X, q, seed, kernel="rbf", sigma_factor=1.0, degree=3, coef0=1.0):
    """Kernel PCA. Gaussian bandwidth sigma = sigma_factor * median distance,
    k(x, x') = exp(-|x - x'|^2 / (2 sigma^2)). Out-of-sample points are
    centred with the TRAINING statistics (scikit-learn does this)."""
    from sklearn.decomposition import KernelPCA

    kw = {"n_components": q, "kernel": kernel, "eigen_solver": "dense", "random_state": seed}
    info = {"kernel": kernel}
    if kernel == "rbf":
        sigma = sigma_factor * median_distance(X)
        kw["gamma"] = 1.0 / (2.0 * sigma ** 2)
        info["sigma"] = sigma
    elif kernel == "poly":
        kw.update(degree=degree, coef0=coef0)
    m = KernelPCA(**kw).fit(X)
    info["eigenvalues"] = m.eigenvalues_
    info["model"] = m
    return Embedding(m.transform(X), m.transform, info)


# ---------------------------------------------------------------------------
# Distance-based spectral methods
# ---------------------------------------------------------------------------

def _classical_mds(D2, q):
    """Classical MDS from squared distances; returns Y, eigvals, eigvecs, row means."""
    n = D2.shape[0]
    row_mean = D2.mean(axis=1)
    B = -0.5 * (D2 - row_mean[:, None] - row_mean[None, :] + row_mean.mean())
    lam, V = eigh(B)
    lam, V = lam[::-1], V[:, ::-1]
    lam_q = np.clip(lam[:q], 0, None)
    Y = V[:, :q] * np.sqrt(lam_q)
    return Y, lam, V[:, :q], row_mean


def _mds(X, q, seed):
    """Classical (Torgerson) MDS on Euclidean distances, with Gower's
    out-of-sample formula y(x) = 1/2 Lambda^{-1/2} V^T (mean_row(D2) - delta_x)."""
    D2 = cdist(X, X, "sqeuclidean")
    Y, lam, Vq, row_mean = _classical_mds(D2, q)
    inv_sqrt = 1.0 / np.sqrt(np.clip(lam[:q], 1e-12, None))

    def transform(Xn):
        delta = cdist(Xn, X, "sqeuclidean")
        return 0.5 * (row_mean[None, :] - delta) @ Vq * inv_sqrt

    neg = lam[lam < 0]
    return Embedding(Y, transform, {"eigenvalues": lam,
                                    "negative_mass": float(-neg.sum() / np.abs(lam).sum())})


def _isomap(X, q, seed, n_neighbors=10):
    from sklearn.manifold import Isomap

    info = {"graph_components": knn_graph_components(X, n_neighbors)}
    with warnings.catch_warnings():
        # a disconnected k-NN graph is expected for small k; it is recorded in info instead
        warnings.filterwarnings("ignore", message="The number of connected components")
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        m = Isomap(n_neighbors=n_neighbors, n_components=q).fit(X)
    info["model"] = m
    # Eigenvalues of the geodesic kernel, incl. negative ones (non-Euclidean diagnostic)
    G2 = m.dist_matrix_ ** 2
    _, lam, _, _ = _classical_mds(G2, q)
    info["eigenvalues"] = lam
    info["negative_mass"] = float(-lam[lam < 0].sum() / np.abs(lam).sum())
    return Embedding(m.embedding_, m.transform, info)


# ---------------------------------------------------------------------------
# Local graph methods
# ---------------------------------------------------------------------------

def _lle(X, q, seed, n_neighbors=10, reg=1e-3):
    from sklearn.manifold import LocallyLinearEmbedding

    info = {"graph_components": knn_graph_components(X, n_neighbors)}
    m = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=q, reg=reg,
                               method="standard", eigen_solver="dense", random_state=seed).fit(X)
    info["reconstruction_error"] = m.reconstruction_error_
    info["model"] = m
    return Embedding(m.embedding_, m.transform, info)


def _le(X, q, seed, n_neighbors=10, tau_factor=1.0):
    """Laplacian eigenmaps (Belkin and Niyogi 2003).

    Symmetrised k-NN graph; heat weights exp(-d^2 / tau) with
    tau = tau_factor * median squared neighbour distance, or binary weights
    if tau_factor is None. Solves L v = lambda D v and drops the constant
    eigenvector. Out-of-sample: Nystrom extension (Bengio et al. 2004),
    y_a(x) = sum_j w(x, x_j) y_aj / ((1 - lambda_a) d(x)).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    dist, ind = dist[:, 1:], ind[:, 1:]
    n = X.shape[0]
    tau = None if tau_factor is None else tau_factor * float(np.median(dist ** 2))

    def weights(d):
        return np.ones_like(d) if tau is None else np.exp(-d ** 2 / tau)

    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), n_neighbors), ind.ravel()] = weights(dist).ravel()
    W = np.maximum(W, W.T)
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    lam, V = eigh(L, np.diag(deg), subset_by_index=[0, q])
    Y = V[:, 1:q + 1]
    lam_q = lam[1:q + 1]

    def transform(Xn):
        d, j = nn.kneighbors(Xn, n_neighbors=n_neighbors)
        w = weights(d)
        num = np.einsum("mk,mkq->mq", w, Y[j])
        return num / ((1.0 - lam_q)[None, :] * w.sum(axis=1, keepdims=True))

    info = {"eigenvalues": lam, "tau": tau,
            "graph_components": int(connected_components(W > 0, directed=False)[0])}
    return Embedding(Y, transform, info)


# ---------------------------------------------------------------------------
# Neighbour embeddings
# ---------------------------------------------------------------------------

def _tsne(X, q, seed, perplexity=30.0):
    from sklearn.manifold import TSNE

    method = "barnes_hut" if q <= 3 else "exact"
    m = TSNE(n_components=q, perplexity=perplexity, init="pca", learning_rate="auto",
             method=method, random_state=seed)
    Y = m.fit_transform(X)
    return Embedding(Y, None, {"kl_divergence": float(m.kl_divergence_), "n_iter": m.n_iter_})


def _umap(X, q, seed, n_neighbors=15, min_dist=0.1):
    try:
        import umap
    except ImportError as e:
        raise ImportError("UMAP needs `pip install umap-learn`") from e
    import inspect

    kw = dict(n_neighbors=n_neighbors, min_dist=min_dist, n_components=q, random_state=seed)
    # umap-learn >= 0.6: keep the classic algorithm of McInnes et al. (2018), as in the notes
    if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:
        kw["compatibility_layout"] = True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value")   # seeding forces one thread
        m = umap.UMAP(**kw).fit(X)
    return Embedding(m.embedding_, m.transform, {"model": m})


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

def _ae(X, q, seed, hidden=(512, 128), weight_decay=0.0, epochs=100, lr=1e-3,
        batch_size=128, X_val=None, patience=None):
    """Fully connected autoencoder d -> hidden -> q -> reversed(hidden) -> d.

    ReLU hidden layers, linear code, sigmoid output (pixels in [0, 1]),
    squared-error loss, Adam. Records train (and validation) reconstruction
    error per epoch in info["history"]; with `patience`, stops early on the
    validation loss and restores the best weights.
    """
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise ImportError("The autoencoder needs `pip install torch`") from e

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]

    def mlp(sizes, last):
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers[-1] = last
        return nn.Sequential(*layers)

    enc = mlp([d, *hidden, q], nn.Identity())
    dec = mlp([q, *reversed(hidden), d], nn.Sigmoid())
    model = nn.Sequential(enc, dec)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Xv = None if X_val is None else torch.as_tensor(X_val, dtype=torch.float32)

    def recon_error(T):
        model.eval()
        with torch.no_grad():
            return float(((model(T) - T) ** 2).sum(1).mean())

    history = {"train": [], "val": []}
    best, best_state, wait = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for idx in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // batch_size)):
            xb = Xt[idx]
            loss = ((model(xb) - xb) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        history["train"].append(recon_error(Xt))
        if Xv is not None:
            v = recon_error(Xv)
            history["val"].append(v)
            if patience is not None:
                if v < best - 1e-6:
                    best, wait = v, 0
                    best_state = {k: t.clone() for k, t in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break
    if best_state is not None:
        model.load_state_dict(best_state)

    def transform(Xn):
        enc.eval()
        with torch.no_grad():
            return enc(torch.as_tensor(Xn, dtype=torch.float32)).numpy().astype(np.float64)

    def reconstruct(Xn):
        model.eval()
        with torch.no_grad():
            return model(torch.as_tensor(Xn, dtype=torch.float32)).numpy().astype(np.float64)

    return Embedding(transform(X), transform,
                     {"history": history, "reconstruct": reconstruct, "epochs_run": len(history["train"])})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {"pca": _pca, "kpca": _kpca, "mds": _mds, "isomap": _isomap, "lle": _lle,
             "le": _le, "tsne": _tsne, "umap": _umap, "ae": _ae}


def fit(name, X, q=2, params=None, seed=0, **extra):
    """Fit method `name` on X (n, d) and return an Embedding of dimension q.

    `params` are the hyperparameters from config.GRIDS; `extra` passes
    method-specific arguments (e.g. X_val, epochs, patience for the autoencoder).
    """
    if name not in _DISPATCH:
        raise ValueError(f"unknown method {name!r}; choose from {METHODS}")
    X = np.asarray(X, dtype=np.float64)
    return _DISPATCH[name](X, q, seed, **(params or {}), **extra)
