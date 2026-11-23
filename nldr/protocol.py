"""The comparison protocol of the vault note "00 Experiment overview".

Track B (in-sample, every method):
    for each configuration: fit on X_val, score J on X_val;
    select the best; fit it on the independent X_te; report every criterion on X_te.

Track A (out-of-sample, methods with a map):
    for each configuration: fit on X_tr, embed X_val with the learned map, score J_oos;
    select the best; embed X_te with that same fitted map; report on X_te.

Every embedding is cached (results/embeddings/), so re-running only computes
what is missing. A configuration that crashes gets J = NaN and is skipped.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import cache, config, metrics, methods

NO_MAP = {"tsne"}   # methods without an out-of-sample map: excluded from Track A


def _params_str(p):
    return ", ".join(f"{k}={v}" for k, v in (p or {}).items()) or "-"


def _safe(fn, what):
    try:
        return fn()
    except Exception as err:   # a bad configuration should not stop the whole run
        warnings.warn(f"{what} failed: {type(err).__name__}: {err}")
        return None


def track_b(method, X, y, split, q, rho_val, rho_te, grid=None, seed=0,
            fit_kwargs=None, Ks=None):
    """Select on X_val (in-sample J), refit and report on X_te. Returns a dict."""
    grid = config.GRIDS[method] if grid is None else grid
    fit_kwargs = fit_kwargs or {}
    Ks = config.K_REPORT if Ks is None else Ks
    X_val, X_te, y_te = X[split.val], X[split.te], y[split.te]

    sel = []
    for p in grid:
        Y = _safe(lambda: cache.embedding(
            method, p, split.seed, seed, q, tag="B-val",
            compute=lambda: methods.fit(method, X_val, q, p, seed, **fit_kwargs).Y),
            f"{method} [{_params_str(p)}] on X_val")
        J = np.nan if Y is None or not np.isfinite(Y).all() else \
            metrics.auc_log(metrics.rnx_curve(rho_val, metrics.rank_matrix(Y)))
        sel.append(J)
    if np.all(np.isnan(sel)):
        return None
    best = grid[int(np.nanargmax(sel))]

    Y_te, seconds = cache.embedding(
        method, best, split.seed, seed, q, tag="B-te", return_seconds=True,
        compute=lambda: methods.fit(method, X_te, q, best, seed, **fit_kwargs).Y)
    res = metrics.evaluate(X_te, Y_te, y_te, Ks=Ks, rho=rho_te)
    return {"method": method, "split": split.seed, "q": q, "params": _params_str(best),
            "J_val_best": float(np.nanmax(sel)), "selection": list(zip(map(_params_str, grid), sel)),
            "seconds": seconds, "Y": Y_te, **res}


def track_a(method, X, y, split, q, grid=None, seed=0, fit_kwargs=None):
    """Fit on X_tr, select by out-of-sample J on X_val, report on X_te. Returns a dict."""
    if method in NO_MAP:
        return None
    grid = config.GRIDS[method] if grid is None else grid
    fit_kwargs = fit_kwargs or {}
    X_tr, X_val, X_te = X[split.tr], X[split.val], X[split.te]
    n_tr, n_val = len(split.tr), len(split.val)

    def fit_all(p):
        emb = methods.fit(method, X_tr, q, p, seed, **fit_kwargs)
        return np.vstack([emb.Y, emb.transform(X_val), emb.transform(X_te)])

    stacks, sel = [], []
    for p in grid:
        Z = _safe(lambda: cache.embedding(method, p, split.seed, seed, q, tag="A",
                                          compute=lambda: fit_all(p)),
                  f"{method} [{_params_str(p)}] fit on X_tr")
        stacks.append(Z)
        if Z is None or not np.isfinite(Z).all():
            sel.append(np.nan)
            continue
        Y_tr, Y_val = Z[:n_tr], Z[n_tr:n_tr + n_val]
        sel.append(metrics.auc_log(metrics.rnx_curve_oos(X_tr, Y_tr, X_val, Y_val)))
    if np.all(np.isnan(sel)):
        return None
    b = int(np.nanargmax(sel))
    Z = stacks[b]
    Y_tr, Y_te = Z[:n_tr], Z[n_tr + n_val:]
    rnx = metrics.rnx_curve_oos(X_tr, Y_tr, X_te, Y_te)
    return {"method": method, "split": split.seed, "q": q, "params": _params_str(grid[b]),
            "J_val_best": float(sel[b]), "selection": list(zip(map(_params_str, grid), sel)),
            "J_oos": metrics.auc_log(rnx), "rnx_oos": rnx,
            "knn_acc_oos": metrics.knn_accuracy_oos(Y_tr, y[split.tr], Y_te, y[split.te],
                                                    k=config.KNN_CLASSIFIER_K),
            "Y_tr": Y_tr, "Y_te": Y_te}


ARRAY_KEYS = ("Y", "Y_tr", "Y_te", "rnx", "rnx_oos", "selection")


def to_table(rows):
    """Flat DataFrame of the scalar results (one row per method and split)."""
    import pandas as pd

    return pd.DataFrame([{k: v for k, v in r.items() if k not in ARRAY_KEYS}
                         for r in rows if r is not None])


def summarise(table, columns, order=None):
    """Mean ± s.d. over splits, one row per method, as strings 'm ± s'."""
    g = table.groupby("method")[columns]
    mean, sd = g.mean(), g.std(ddof=1).fillna(0.0)
    out = mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)
    out.insert(0, "n_splits", g.size())
    if order is not None:
        out = out.reindex([m for m in order if m in out.index])
    return out
