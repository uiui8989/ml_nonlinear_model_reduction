"""Loading MNIST and drawing the protocol splits.

Pool for the splits: the official 60,000 training images. The official
10,000 test images are kept aside and never touched by the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Settings shared by everyone (change only through a pull request) -------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"   # MNIST is saved here
N_PER_SPLIT = 2000              # size of each of X_tr, X_val, X_te
SPLIT_SEEDS = list(range(5))    # one split per seed; same seed -> same split


def load_mnist(data_home=None):
    """Return X (70000, 784) float64 in [0, 1] and y (70000,) int.

    Rows 0..59999 are the official training set, 60000..69999 the test set.
    The first call downloads about 50 MB from OpenML and saves a compact copy
    as data/mnist.npz; later calls load that copy in about a second.
    """
    data_home = DATA_DIR if data_home is None else data_home
    data_home = Path(data_home)
    cached = data_home / "mnist.npz"
    if cached.exists():
        with np.load(cached) as f:
            return f["X"].astype(np.float64) / 255.0, f["y"].astype(int)

    from sklearn.datasets import fetch_openml

    kw = dict(version=1, return_X_y=True, as_frame=False, data_home=str(data_home))
    try:
        X, y = fetch_openml("mnist_784", parser="pandas", **kw)
    except Exception as err:  # e.g. scikit-learn's pandas parser vs pandas >= 3
        print(f"fast ARFF reader failed ({type(err).__name__}); using the slower "
              "pure-Python reader, this takes a minute or two...")
        X, y = fetch_openml("mnist_784", parser="liac-arff", **kw)
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y).astype(int)
    data_home.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, X=X, y=y)
    return X.astype(np.float64) / 255.0, y


@dataclass(frozen=True)
class Split:
    """Indices (into the full array) of one repetition of the protocol."""
    seed: int
    tr: np.ndarray
    val: np.ndarray
    te: np.ndarray


def make_splits(n=None, seeds=None, pool_size=60000):
    """Disjoint X_tr, X_val, X_te of equal size n, one Split per seed.

    Equal sizes matter: neighbour-type hyperparameters (k_nn, perplexity)
    scale with n. Same seed -> same split for everyone in the group.
    """
    n = N_PER_SPLIT if n is None else n
    seeds = SPLIT_SEEDS if seeds is None else seeds
    if 3 * n > pool_size:
        raise ValueError(f"3n = {3 * n} exceeds the pool of {pool_size} images")
    splits = []
    for s in seeds:
        perm = np.random.default_rng(s).permutation(pool_size)
        splits.append(Split(s, perm[:n], perm[n:2 * n], perm[2 * n:3 * n]))
    return splits


def class_subset(y, digit, n, seed=0, pool_size=60000):
    """Indices of n random training images of a single digit (for E5)."""
    idx = np.flatnonzero(y[:pool_size] == digit)
    return np.random.default_rng(seed).choice(idx, size=min(n, idx.size), replace=False)
