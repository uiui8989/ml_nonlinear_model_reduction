"""Standard figures, so every experiment looks the same."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import config


def save(fig, name):
    """Save to figures/<name>.png and .pdf (for the report and the vault)."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def rnx_curves(curves, ax=None, title=None):
    """R_NX(K) against log K. `curves` maps a label to (rnx array) or (rnx, J)."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    for label, c in curves.items():
        rnx, J = (c if isinstance(c, tuple) else (c, None))
        K = np.arange(1, len(rnx) + 1)
        ax.semilogx(K, rnx, label=label if J is None else f"{label} ({J:.3f})")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("$K$ (log scale)")
    ax.set_ylabel("$R_{NX}(K)$")
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.0)
    if title:
        ax.set_title(title)
    if len(curves) > 4:
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    else:
        ax.legend(fontsize=8)
    return ax


def embedding_grid(embeddings, labels, ncols=3, s=2):
    """Scatter plots of 2-D embeddings coloured by digit. `embeddings`: label -> (n, 2)."""
    n = len(embeddings)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax, (name, Y) in zip(axes.flat, embeddings.items()):
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=s, vmin=-0.5, vmax=9.5)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes, ticks=range(10), shrink=0.6, label="digit")
    return fig


def digit_images(X, idx, ncols=10, titles=None):
    """Show MNIST images X[idx] (rows of 784 pixels) in a grid."""
    idx = list(idx)
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows * 1.1), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < len(idx):
            ax.imshow(X[idx[k]].reshape(28, 28), cmap="gray_r")
            if titles is not None:
                ax.set_title(titles[k], fontsize=7)
    return fig


def shepard(X, Y, ax=None, max_pairs=20000, seed=0, title=None):
    """Shepard diagram: embedding distances against input distances (subsampled)."""
    from scipy.spatial.distance import pdist

    ax = ax or plt.subplots(figsize=(4, 4))[1]
    dX, dY = pdist(X), pdist(Y)
    idx = np.random.default_rng(seed).choice(dX.size, min(max_pairs, dX.size), replace=False)
    ax.scatter(dX[idx], dY[idx] * (dX[idx].mean() / dY[idx].mean()), s=1, alpha=0.2)
    lim = dX[idx].max()
    ax.plot([0, lim], [0, lim], color="0.4", lw=0.8)
    ax.set_xlabel("input distance")
    ax.set_ylabel("embedding distance (rescaled)")
    if title:
        ax.set_title(title)
    return ax
