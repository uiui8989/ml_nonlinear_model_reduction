# NLDR on MNIST

Group project, MATH-412 Statistical Machine Learning (EPFL).
We compare nonlinear dimensionality reduction methods on MNIST.

## Layout

```
nldr/                          shared code (change only through a pull request)
  data.py                      load MNIST, make the splits (n, seeds)
  metrics.py                   scores: R_NX, AUC, trustworthiness, k-NN accuracy, ...
notebooks/
  01_baselines.ipynb           run first: data check + PCA baselines
  02_yifei_spectral.ipynb      kernel PCA, MDS, Isomap          (Yifei)
  03_jia_local.ipynb           LLE, Laplacian eigenmaps         (Jia)
  04_sophia_neighbour.ipynb    t-SNE, UMAP, autoencoder         (Sophia)
results/                       CSV tables  (baseline_*, 02_*, 03_*, 04_*)
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

## Using the shared code

```python
from nldr import data, metrics
X, y = data.load_mnist()
s = data.make_splits()[0]                 # split 0: indices s.tr, s.val, s.te
Y = ...                                   # your embedding of X[s.te], shape (n, q)
res = metrics.evaluate(X[s.te], Y, y[s.te])
res["J"]                                  # compare with results/baseline_pca_split0.csv
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

1. Always load data with `data.load_mnist()` and `data.make_splits()`, so everyone uses the same splits.
2. Score every method with `metrics`, so the numbers are comparable.
3. Edit only your own notebook; save outputs with your prefix (`02_`, `03_`, `04_`).
4. Never commit `data/` or `results/embeddings/`.
