"""Quality criteria for embeddings (see the vault note "Quality criteria").

All functions take samples as ROWS: X has shape (n, d), Y has shape (n, q).

Ranks: rho[i, j] is the rank of x_j among the neighbours of x_i (1 = nearest,
0 = i itself), r[i, j] the same in the embedding. Ties are broken by index
(stable sort), so results are deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Ranks and the co-ranking matrix
# ---------------------------------------------------------------------------

def rank_matrix(Z):
    """(n, n) int32 matrix of neighbour ranks; diagonal = 0, others 1..n-1."""
    Z = np.asarray(Z, dtype=np.float64)
    D = cdist(Z, Z)
    np.fill_diagonal(D, -np.inf)          # self always first, even with duplicates
    order = np.argsort(D, axis=1, kind="stable")
    n = D.shape[0]
    R = np.empty((n, n), dtype=np.int32)
    np.put_along_axis(R, order, np.broadcast_to(np.arange(n, dtype=np.int32), (n, n)), axis=1)
    return R


def coranking_matrix(rho, r):
    """Q[k-1, l-1] = #{(i, j), i != j : rho_ij = k, r_ij = l}, shape (n-1, n-1).

    Memory is (n-1)^2 integers: use for small n or diagnostics only; the
    criteria below never form it.
    """
    n = rho.shape[0]
    off = ~np.eye(n, dtype=bool)
    flat = (rho[off].astype(np.int64) - 1) * (n - 1) + (r[off].astype(np.int64) - 1)
    return np.bincount(flat, minlength=(n - 1) ** 2).reshape(n - 1, n - 1)


# ---------------------------------------------------------------------------
# Neighbourhood preservation: Q_NX, R_NX, AUC_log
# ---------------------------------------------------------------------------

def qnx_curve(rho, r):
    """Q_NX(K) for K = 1..n-1 (index K-1).

    A pair (i, j) lies in the upper-left K x K block of the co-ranking matrix
    iff max(rho_ij, r_ij) <= K, so a histogram of that maximum plus a
    cumulative sum gives every block sum at once, in O(n^2) memory.
    """
    n = rho.shape[0]
    m = np.maximum(rho, r)
    counts = np.bincount(m.ravel(), minlength=n)[1:]   # drop m = 0 (the diagonal)
    K = np.arange(1, n)
    return np.cumsum(counts) / (n * K)


def rnx_curve(rho, r):
    """R_NX(K) for K = 1..n-2 (index K-1): 0 = random embedding, 1 = perfect."""
    n = rho.shape[0]
    q = qnx_curve(rho, r)[: n - 2]
    K = np.arange(1, n - 1)
    return ((n - 1) * q - K) / (n - 1 - K)


def auc_log(rnx):
    """Area under R_NX with K on a log scale: weights 1/K, normalised."""
    rnx = np.asarray(rnx, dtype=np.float64)
    w = 1.0 / np.arange(1, rnx.size + 1)
    return float(np.sum(w * rnx) / np.sum(w))


# ---------------------------------------------------------------------------
# Folding versus tearing: trustworthiness and continuity
# ---------------------------------------------------------------------------

def trustworthiness_continuity(rho, r, K):
    """T(K) and C(K) (Venna and Kaski), valid for K < n/2.

    T penalises false neighbours (hard intrusions) by their input rank,
    C penalises missed neighbours (hard extrusions) by their embedding rank.
    """
    n = rho.shape[0]
    if not 1 <= K < n / 2:
        raise ValueError("need 1 <= K < n/2")
    in_X = (rho >= 1) & (rho <= K)
    in_Y = (r >= 1) & (r <= K)
    norm = 2.0 / (n * K * (2 * n - 3 * K - 1))
    T = 1.0 - norm * np.sum((rho - K)[in_Y & ~in_X])
    C = 1.0 - norm * np.sum((r - K)[in_X & ~in_Y])
    return float(T), float(C)


# ---------------------------------------------------------------------------
# Out-of-sample neighbourhood preservation (training set as reference)
# ---------------------------------------------------------------------------

def _cross_ranks(A, B):
    """rank[l, i] of B_i among the neighbours of A_l (1 = nearest), shape (m, n)."""
    D = cdist(np.asarray(A, float), np.asarray(B, float))
    order = np.argsort(D, axis=1, kind="stable")
    m, n = D.shape
    R = np.empty((m, n), dtype=np.int32)
    np.put_along_axis(R, order, np.broadcast_to(np.arange(1, n + 1, dtype=np.int32), (m, n)), axis=1)
    return R


def rnx_curve_oos(X_tr, Y_tr, X_te, Y_te):
    """Out-of-sample R_NX(K), K = 1..n-1, with the n training points as reference.

    For each test point, compare its K nearest TRAINING points in input space
    with its K nearest training points in the embedding. Chance level is
    K/n, so R = (n Q - K) / (n - K).
    """
    rho = _cross_ranks(X_te, X_tr)
    r = _cross_ranks(Y_te, Y_tr)
    m, n = rho.shape
    counts = np.bincount(np.maximum(rho, r).ravel(), minlength=n + 1)[1:]
    K = np.arange(1, n + 1)
    q = np.cumsum(counts) / (m * K)
    K, q = K[:-1], q[:-1]
    return (n * q - K) / (n - K)


# ---------------------------------------------------------------------------
# Global structure and label-based criteria
# ---------------------------------------------------------------------------

def spearman_distances(X, Y, max_pairs=500_000, seed=0):
    """Spearman correlation between input and embedding pairwise distances."""
    dX, dY = pdist(X), pdist(Y)
    if dX.size > max_pairs:
        idx = np.random.default_rng(seed).choice(dX.size, max_pairs, replace=False)
        dX, dY = dX[idx], dY[idx]
    return float(spearmanr(dX, dY).statistic)


def centroid_correlation(X, Y, labels):
    """Spearman correlation of the distances between class means in X and in Y."""
    classes = np.unique(labels)
    cX = np.stack([X[labels == c].mean(0) for c in classes])
    cY = np.stack([Y[labels == c].mean(0) for c in classes])
    return float(spearmanr(pdist(cX), pdist(cY)).statistic)


def knn_accuracy(Y, labels, k=5, folds=5, seed=0):
    """Cross-validated k-NN accuracy inside the embedding (in-sample)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(cross_val_score(KNeighborsClassifier(k), Y, labels, cv=cv).mean())


def knn_accuracy_oos(Y_tr, y_tr, Y_te, y_te, k=5):
    """k-NN trained on the training embedding, tested on embedded test points."""
    from sklearn.neighbors import KNeighborsClassifier

    return float(KNeighborsClassifier(k).fit(Y_tr, y_tr).score(Y_te, y_te))


def cluster_scores(Y, labels, seed=0):
    """Silhouette with the true labels, and ARI / NMI of k-means on Y."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    km = KMeans(n_clusters=np.unique(labels).size, n_init=10, random_state=seed).fit(Y)
    return {
        "silhouette": float(silhouette_score(Y, labels)),
        "ari": float(adjusted_rand_score(labels, km.labels_)),
        "nmi": float(normalized_mutual_info_score(labels, km.labels_)),
    }


# ---------------------------------------------------------------------------
# One call for the standard table
# ---------------------------------------------------------------------------

def evaluate(X, Y, labels=None, Ks=(5, 10, 50), rho=None):
    """All in-sample criteria of one embedding, as a flat dict.

    Pass `rho = rank_matrix(X)` when scoring many embeddings of the same X:
    the input ranks are the expensive part and do not depend on the method.
    """
    rho = rank_matrix(X) if rho is None else rho
    r = rank_matrix(Y)
    rnx = rnx_curve(rho, r)
    out = {"J": auc_log(rnx), "rnx": rnx}
    for K in Ks:
        out[f"T{K}"], out[f"C{K}"] = trustworthiness_continuity(rho, r, K)
    out["spearman"] = spearman_distances(X, Y)
    if labels is not None:
        out["centroid_corr"] = centroid_correlation(X, Y, labels)
        out["knn_acc"] = knn_accuracy(Y, labels)
        out.update(cluster_scores(Y, labels))
    return out
