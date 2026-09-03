"""Plotting helpers for price profiles and convergence studies.

The Agg backend is used so figures render headless and can be written
straight to PNG files from the CLI or scripts/convergence.py.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_order(ax, xs, errors, order, xlabel, label):
    """Log-log error vs step size with a reference line of slope ``order``.

    ax: target axes, xs: step sizes (dt or h), errors: measured errors,
    order: theoretical order whose slope is drawn through the first point.
    """
    xs = np.asarray(xs, dtype=float)
    errors = np.asarray(errors, dtype=float)
    ax.loglog(xs, errors, "o-", label=label)
    span = np.logspace(np.log10(xs[0]), np.log10(xs[-1]), 50)
    ref = errors[0] * (span / xs[0]) ** order
    ax.loglog(span, ref, "--", color="gray",
              label=f"O({xlabel.split()[0]}^{order:g})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative error")
    ax.legend()


def price_profile(S_nodes, V_fd, V_bs, S0, K, kind, path):
    """Save a figure of the finite-difference price curve vs Black-Scholes.

    S_nodes: spot values on the grid, V_fd: FD price vector on the same
    nodes, V_bs: Black-Scholes price evaluated at those spots, S0: quoted
    spot marked on the curve, path: output PNG file.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(S_nodes, V_fd, "o", ms=3, label="finite difference")
    ax.plot(S_nodes, V_bs, "-", lw=1.4, label="Black-Scholes")
    ax.axvline(S0, color="gray", lw=0.8, ls=":")
    ax.plot([S0], [np.interp(S0, S_nodes, V_fd)], "o", color="C3",
            label=f"S0 = {S0:g}")
    ax.set_xlabel("spot S")
    ax.set_ylabel(f"{kind} price")
    ax.set_title(f"{kind} price profile, K = {K:g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
