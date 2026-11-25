"""Shared code for the NLDR group project (MATH-412): data loading, metrics and
freezing hyperparameters.

Convention: arrays follow scikit-learn, one sample per ROW.
The Obsidian notes use one sample per COLUMN, so X_code = X_notes.T.
"""

from . import data, metrics, selection  # noqa: F401

__all__ = ["data", "metrics", "selection"]
