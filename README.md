# NLDR on MNIST — shared code


Group project, MATH-412 (Statistical Machine Learning, EPFL). Plans and results live in the Obsidian vault (`Project/Notes`); this folder holds the code.

## Layout

```
nldr/              shared package, imported by every notebook
  config.py        protocol: n per split, S, seeds, q, hyperparameter grids
  data.py          load_mnist(), make_splits(), class_subset()
  methods.py       fit(name, X, q, params, seed) -> Embedding(Y, transform, info), nine methods
  metrics.py       ranks, co-ranking matrix, R_NX, AUC_log, T/C, out-of-sample R_NX,
                   Spearman, centroid correlation, k-NN accuracy, silhouette, ARI/NMI, evaluate()
  plotting.py      R_NX curves, embedding grids, digit images, Shepard diagrams, save()
  cache.py         cache embeddings in results/embeddings/ so expensive runs happen once
tests/             unit tests (incl. the worked example of the vault note "Quality criteria")
notebooks/         00_data, E1_baselines, ... one notebook per experiment of the vault
results/           CSV tables, cached embeddings
figures/           PNG + PDF, for the report and the vault
```

## Setup

Use a dedicated conda environment, not `base`: UMAP needs NumPy 2, which breaks several packages that ship with Anaconda's base environment.

```bash
conda env create -f environment.yml      # once
conda activate nldr
python -m ipykernel install --user --name nldr --display-name "Python (nldr)"   # once, for Jupyter
python -m pytest -q                      # all tests should pass
```

In Jupyter, pick the kernel **Python (nldr)** for every notebook.

## Conventions

- **Samples are rows** (scikit-learn): `X` has shape `(n, 784)`, an embedding `Y` has shape `(n, q)`. The vault notes use columns, so `X` here is $X^\top$ there.
- Pixels are scaled to $[0,1]$; nothing else is done to the input unless an experiment says so.
- All sizes, seeds and grids come from `nldr/config.py`. Change them there, never inside a notebook.
- Splits: `data.make_splits()` draws disjoint `tr`, `val`, `te` index sets of equal size from the 60,000 official training images; split *s* is identical for everyone.

## Workflow

1. Open a notebook in `notebooks/`. The first cell turns on `autoreload`, so edits to `nldr/*.py` are picked up without restarting.
2. Run it top to bottom. Figures go to `figures/`, tables to `results/`.
3. The last cell prints a summary; paste it into `MNIST — Results log` in the vault.

## Adding or changing a method

Write `_name(X, q, seed, **params) -> Embedding` in `nldr/methods.py`, register it in `_DISPATCH`, add its grid to `config.GRIDS`, and add a test in `tests/`. Every notebook then works with it.

## Notes

- `n x n` matrices: 32 MB at n = 2000, 200 MB at n = 5000. Kernel PCA, MDS, Isomap, LLE and Laplacian eigenmaps use dense solvers; keep n ≤ 5000 for them.
- Location: keep this folder (and its Git repository) **outside iCloud Drive**; iCloud sync can corrupt `.git`. If "Desktop & Documents Folders" is turned on in iCloud settings, the Desktop is synced too, so use e.g. `~/code/nldr` instead.
- MNIST is saved once as `data/mnist.npz`; delete it to force a fresh download.
- VS Code: `.vscode/settings.json` points to the `nldr` interpreter and runs notebooks from their own folder. It is machine-specific and not committed.
