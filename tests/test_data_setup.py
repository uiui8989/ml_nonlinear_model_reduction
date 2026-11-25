"""Tests for the data setup: splits, final holdout, freezing."""

import numpy as np
import pandas as pd
import pytest

from nldr import data, selection


def test_splits_disjoint_fixed_and_in_pool():
    a, b = data.make_splits(), data.make_splits()
    for s, t in zip(a, b):
        assert np.array_equal(s.tr, t.tr) and np.array_equal(s.te, t.te)   # fixed
        idx = np.concatenate([s.tr, s.val, s.te])
        assert len(np.unique(idx)) == 3 * data.N_PER_SPLIT                 # disjoint
        assert idx.max() < data.POOL_SIZE                                  # training pool only


def test_holdout_locked_by_default():
    with pytest.raises(PermissionError):
        data.final_holdout_chunks()


def test_holdout_chunks():
    chunks = data.final_holdout_chunks(unlock=True)
    assert len(chunks) == len(data.SPLIT_SEEDS)
    assert all(len(c) == data.N_PER_SPLIT for c in chunks)
    idx = np.concatenate(chunks)
    assert len(np.unique(idx)) == idx.size                                  # disjoint
    assert idx.min() >= data.POOL_SIZE and idx.max() < data.POOL_SIZE + data.HOLDOUT_SIZE
    again = data.final_holdout_chunks(unlock=True)
    assert all(np.array_equal(c, d) for c, d in zip(chunks, again))         # fixed


def test_select_frozen_uses_mean_over_splits(tmp_path):
    # config "a" wins 3 of 4 splits by a little, "b" wins one by a lot: the mean picks "b"
    rows = [(s, "iso", c, 2, v) for s, (va, vb) in enumerate([(.51, .50), (.51, .50),
                                                              (.51, .50), (.10, .90)])
            for c, v in (("a", va), ("b", vb))]
    runs = pd.DataFrame(rows, columns=["split", "method", "config", "q", "J_A_val"])
    fz = selection.select_frozen(runs, "J_A_val", "A", params={("iso", "b"): {"k": 7}})
    assert fz.loc[0, "config"] == "b"
    path = tmp_path / "frozen.csv"
    selection.save_frozen(fz, path)
    selection.save_frozen(fz, path)                                         # replace, no duplicate
    assert len(selection.load_frozen(path)) == 1
    assert selection.frozen_params("iso", 2, "A", path) == {"k": 7}


def test_select_frozen_rejects_incomplete_runs():
    runs = pd.DataFrame({"split": [0, 1, 0], "method": "m", "config": ["a", "a", "b"],
                         "q": 2, "J": [1., 1., 2.]})
    with pytest.raises(ValueError):
        selection.select_frozen(runs, "J", "B")


def test_splits_stored_in_repo():
    data.make_splits()
    assert (data.SPLITS_DIR / f"splits_n{data.N_PER_SPLIT}.npz").exists()
    assert data.check_stored_splits()


def test_non_default_splits_are_not_stored():
    data.make_splits(n=50, seeds=[0])
    assert not (data.SPLITS_DIR / "splits_n50.npz").exists()
