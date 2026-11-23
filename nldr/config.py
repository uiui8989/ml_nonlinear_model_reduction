"""Single place for the experimental protocol: sizes, seeds, grids.

Change values here, not inside notebooks, so every experiment and every
teammate uses the same settings.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # scikit-learn's OpenML cache
RESULTS_DIR = ROOT / "results"    # CSV tables and cached embeddings (.npz)
FIGURES_DIR = ROOT / "figures"    # PNG / PDF for the report and the vault

# --- Protocol --------------------------------------------------------------
N_PER_SPLIT = 2000        # size of each of X_tr, X_val, X_te
N_REPEATS = 5             # S, number of independent split repetitions
SPLIT_SEEDS = list(range(N_REPEATS))
METHOD_SEEDS = [0, 1, 2, 3, 4]   # for non-convex methods (t-SNE, UMAP, AE)
Q_FIGURES = 2             # target dimension for figures
Q_TABLES = 10             # target dimension for tables
K_REPORT = (5, 10, 50)    # neighbourhood sizes reported for T(K), C(K)
KNN_CLASSIFIER_K = 5      # k for k-NN accuracy

# --- Hyperparameter grids (equal budget, about 10 configurations) ----------
# Gaussian bandwidths are multiples of the median pairwise distance.
GRIDS = {
    "pca": [{}],
    "mds": [{}],
    "kpca": [{"kernel": "rbf", "sigma_factor": s} for s in (0.1, 0.3, 1.0, 3.0, 10.0)]
            + [{"kernel": "poly", "degree": p} for p in (2, 3, 5)],
    "isomap": [{"n_neighbors": k} for k in (5, 10, 20, 50, 100)],
    "lle": [{"n_neighbors": k, "reg": r} for k in (5, 10, 20, 50) for r in (1e-3, 1e-1)],
    "le": [{"n_neighbors": k, "tau_factor": t} for k in (5, 10, 20, 50) for t in (1.0, None)],
    "tsne": [{"perplexity": p} for p in (5, 15, 30, 50, 100)],
    "umap": [{"n_neighbors": k, "min_dist": m} for k in (5, 15, 50) for m in (0.0, 0.1, 0.5)],
    "ae": [{"hidden": h, "weight_decay": w} for h in ((256,), (512, 128), (1000, 500, 250))
           for w in (0.0, 1e-4)],
}
