"""Cache embeddings on disk so expensive runs happen once.

    Y = cache.embedding("tsne", {"perplexity": 30}, split_seed=0, method_seed=0, q=2,
                        compute=lambda: methods.fit("tsne", X_tr, 2, {"perplexity": 30}).Y)

The file name encodes method, parameters, split, seed, q and a tag, so
re-running a notebook only recomputes what is missing. The wall-clock time of
the computation is stored with the embedding. Delete results/embeddings/ to
start over.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import config


def _key(method, params, split_seed, method_seed, q, tag):
    p = "_".join(f"{k}={v}" for k, v in sorted((params or {}).items())) or "default"
    p = p.replace("/", "-").replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")
    return f"{method}__{p}__split{split_seed}__seed{method_seed}__q{q}{('__' + tag) if tag else ''}"


def embedding(method, params, split_seed, method_seed, q, compute, tag="", return_seconds=False):
    """Load the cached array if it exists, else call `compute()` and save it.

    With return_seconds=True, also return the time `compute()` took when it
    was first run (NaN for files written by older versions).
    """
    folder = config.RESULTS_DIR / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (_key(method, params, split_seed, method_seed, q, tag) + ".npz")
    if path.exists():
        with np.load(path) as f:
            Y = f["Y"]
            seconds = float(f["seconds"]) if "seconds" in f.files else float("nan")
    else:
        t0 = time.perf_counter()
        Y = np.asarray(compute())
        seconds = time.perf_counter() - t0
        np.savez_compressed(path, Y=Y, seconds=seconds, params=json.dumps(params or {}))
    return (Y, seconds) if return_seconds else Y
