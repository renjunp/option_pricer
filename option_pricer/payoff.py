"""Terminal payoff functions for European call and put options.

The payoff is the boundary/initial data for the finite-difference scheme at
``tau = 0`` and the integrand of the Monte Carlo estimator at maturity.
"""

import numpy as np


def payoff(S, K, kind="call"):
    """European payoff ``max(S-K, 0)`` (call) or ``max(K-S, 0)`` (put).

    S: array-like of spot prices (scalar or ndarray), K: strike > 0.
    kind: "call" or "put".
    """
    S = np.asarray(S, dtype=float)
    if kind == "call":
        return np.maximum(S - K, 0.0)
    if kind == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
