"""Freezing one hyperparameter configuration per (method, q, track).

During development each configuration is scored on every split; the frozen
configuration is the one with the best score AVERAGED over the splits
(not a per-split winner, not a vote). The frozen choices are stored in one
shared file, results/frozen_configs.csv, which the final holdout run reads.

Typical use in a notebook, with `runs` holding one row per
(split, method, config, q) and the selection scores as columns:

    from nldr import selection
    frozen = selection.select_frozen(runs, score="J_A_val", track="A", params=PARAMS)
    selection.save_frozen(frozen)          # adds / replaces your rows only
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FROZEN_FILE = Path(__file__).resolve().parents[1] / "results" / "frozen_configs.csv"
KEY = ["method", "q", "track"]
COLUMNS = KEY + ["config", "params", "score", "score_mean", "score_sd", "n_splits"]


def select_frozen(runs, score, track, params=None, split_col="split",
                  method_col="method", config_col="config", q_col="q"):
    """One row per (method, q): the config with the largest mean `score` over splits.

    runs   : DataFrame, one row per (split, method, config, q).
    score  : column to maximise, e.g. "J_A_val" (Track A) or "J_B_val" (Track B).
    track  : "A" or "B", stored with the choice.
    params : optional dict (method, config) -> keyword dict, stored as JSON so
             anyone can refit the frozen configuration.
    """
    if track not in ("A", "B"):
        raise ValueError("track must be 'A' or 'B'")
    g = runs.groupby([method_col, q_col, config_col], sort=False)
    stats = pd.DataFrame({"score_mean": g[score].mean(), "score_sd": g[score].std(),
                          "n_splits": g[split_col].nunique()}).reset_index()
    n_all = runs[split_col].nunique()
    incomplete = stats[stats["n_splits"] < n_all]
    if len(incomplete):
        raise ValueError("some configurations were not run on every split:\n"
                         f"{incomplete[[method_col, q_col, config_col, 'n_splits']]}")
    best = stats.loc[stats.groupby([method_col, q_col], sort=False)["score_mean"].idxmax()]
    best = best.rename(columns={method_col: "method", q_col: "q", config_col: "config"})
    best["track"] = track
    best["score"] = score
    best["params"] = [json.dumps((params or {}).get((m, c), {}), sort_keys=True)
                      for m, c in zip(best["method"], best["config"])]
    return best[COLUMNS].reset_index(drop=True)


def load_frozen(path=None):
    """The shared table of frozen configurations (empty if none yet)."""
    path = Path(FROZEN_FILE if path is None else path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def save_frozen(frozen, path=None):
    """Add or replace rows of the shared table, keyed by (method, q, track).

    Rows of other methods (other people) are kept unchanged.
    """
    path = Path(FROZEN_FILE if path is None else path)
    old = load_frozen(path)
    if len(old):
        keys = set(map(tuple, frozen[KEY].astype(str).values))
        old = old[[tuple(r) not in keys for r in old[KEY].astype(str).values]]
    parts = [old[COLUMNS]] if len(old) else []
    out = pd.concat(parts + [frozen[COLUMNS]], ignore_index=True)
    out = out.sort_values(KEY).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def frozen_params(method, q, track, path=None):
    """Keyword dict of the frozen configuration, for refitting."""
    t = load_frozen(path)
    row = t[(t["method"] == method) & (t["q"] == q) & (t["track"] == track)]
    if len(row) != 1:
        raise KeyError(f"no frozen configuration for {(method, q, track)}")
    return json.loads(row["params"].iloc[0])
