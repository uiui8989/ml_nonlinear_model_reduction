"""Shared code for the NLDR group project (MATH-412).

Convention: arrays follow scikit-learn, one sample per ROW.
The Obsidian notes use one sample per COLUMN, so X_code = X_notes.T and an
embedding returned here as an (n, q) array is Y_notes.T.
"""

from . import config, data, metrics, methods, plotting, cache, protocol  # noqa: F401

__all__ = ["config", "data", "metrics", "methods", "plotting", "cache", "protocol"]
