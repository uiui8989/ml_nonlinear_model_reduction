# NLDR on MNIST

Group project, MATH-412 Statistical Machine Learning (EPFL).
We compare nonlinear dimensionality reduction methods on MNIST.

## Layout

```
nldr/                          shared code (change only through a pull request)
  data.py                      load MNIST, make the splits, final holdout chunks
  metrics.py                   scores: R_NX, AUC, trustworthiness, k-NN accuracy, ...
  selection.py                 freeze one configuration per method (mean over splits)
notebooks/
  01_baselines.ipynb           run first: data check + PCA baselines
  02_yifei_spectral.ipynb      kernel PCA, MDS, Isomap          (Yifei)
  03_jia_local.ipynb           LLE, Laplacian eigenmaps         (Jia)
  04_sophia_neighbour.ipynb    t-SNE, UMAP, autoencoder         (Sophia)
results/                       CSV tables  (baseline_*, 02_*, 03_*, 04_*, frozen_configs.csv)
figures/                       PNG + PDF   (same prefixes)
tests/                         unit tests for nldr/
environment.yml                conda environment "nldr"
```

## Setup (once)

```bash
git clone git@github.com:uiui8989/ml_nonlinear_model_reduction.git
cd ml_nonlinear_model_reduction
conda env create -f environment.yml
conda activate nldr
python -m pytest -q
```

Then run `notebooks/01_baselines.ipynb` (it downloads MNIST into `data/`).
In Jupyter / VS Code, choose the kernel **nldr**.

## Data setup

```
MNIST (70,000 images, 784 pixels in [0, 1])
├── rows 0–59,999       official training images → DEVELOPMENT (5 fixed splits)
└── rows 60,000–69,999  official test images     → FINAL HOLDOUT (locked)
```

**1. Splits (development).** `data.make_splits()` gives 5 splits (seeds 0–4).
Each split has three disjoint sets of n = 2000 images: `tr`, `val`, `te`.
Same seed → same split for everyone. All sets have the same size because
k, perplexity and `n_neighbors` depend on n.

**2. Tuning, then freezing.** Score every configuration on every split:

| | Track A (out-of-sample) | Track B (in-sample) |
|---|---|---|
| methods | with a map: PCA, kernel PCA, MDS, Isomap, LLE, LE (Nyström), UMAP, AE | all, including t-SNE |
| fit on | `tr` | `val` |
| selection score | map `val` through the fitted map | in-sample score on `val` |

Then freeze **one** configuration per (method, q, track): the best score
**averaged over the 5 splits**, with `selection.select_frozen` +
`selection.save_frozen` → `results/frozen_configs.csv`.
The split test sets `te` are for analysis only (overfitting, hyperparameter
paths); they never choose a configuration and are not in the final table.

**3. Final evaluation (once, together).** `data.final_holdout_chunks(unlock=True)`
gives 5 fixed, disjoint chunks of 2000 official test images; chunk s goes with split s.

| | Track A | Track B |
|---|---|---|
| fit on | `tr` of split s, frozen configuration | chunk s, frozen configuration |
| score | map chunk s, out-of-sample criteria | in-sample criteria on chunk s |

Report mean ± s.d. over the 5 chunks, one table per track.

## Using the shared code

```python
from nldr import data, metrics
X, y = data.load_mnist()
s = data.make_splits()[0]                 # split 0: indices s.tr, s.val, s.te
Y = ...                                   # your embedding of X[s.te], shape (n, q)
res = metrics.evaluate(X[s.te], Y, y[s.te])
res["J"]                                  # compare with results/baseline_pca_split0.csv
```

Freezing your configurations (after running all splits; `runs` has one row per
split, method, config, q and the selection scores as columns):

```python
from nldr import selection
fz = selection.select_frozen(runs, score="J_A_val", track="A", params=PARAMS)
selection.save_frozen(fz)                 # replaces only your own rows
selection.frozen_params("isomap", 2, "A") # keyword dict for refitting
```

## Daily workflow

```bash
git switch main && git pull
git switch -c <name>/<topic>      # e.g. jia/lle-sweep
# work, then
git add <files>
git commit -m "What you did"
git push -u origin <name>/<topic>
```

Open a pull request on GitHub. Your reviewer approves it, then merge.

## Rules

1. Always load data with `data.load_mnist()` and `data.make_splits()` with the default settings, so everyone uses the same splits.
2. Score every method with `metrics`, so the numbers are comparable.
3. Edit only your own notebook; save outputs with your prefix (`02_`, `03_`, `04_`).
4. Never commit `data/` or `results/embeddings/`.
5. Never use rows 60,000–69,999 during development. `final_holdout_chunks` stays locked until every method is in `results/frozen_configs.csv`; no changes to the frozen configurations afterwards.
6. Do not tune random seeds (t-SNE, UMAP, AE): vary them and report the spread.
