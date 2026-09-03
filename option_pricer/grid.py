"""Uniform grid in log-price ``x = ln S`` for the finite-difference scheme.

The grid is centered on ``x0 = ln S0`` so the quoted spot is always a grid
node.  The domain is wide enough that both ``ln S0`` and the strike ``ln K``
sit at least ``beta * sigma * sqrt(T)`` away from the truncated ends; the
option value there differs from its far-field linear asymptote by a Gaussian
tail ~ N(-beta), so with beta = 6 the truncation error is below 1e-8 and
never contaminates the measured convergence rates.
"""

import numpy as np

from .payoff import payoff


class LogGrid:
    """Uniform grid ``x_i = ln S_i`` on [xmin, xmax] with S0 on a node.

    Parameters:
        S0: quoted spot (its log-price is the center node),
        K: strike (the payoff kink must be well inside the domain),
        sigma, T: diffusion scale ``sigma*sqrt(T)`` sets the domain width,
        n_s: number of nodes (forced odd so S0 is the exact middle node),
        beta: domain extends ``beta * sigma * sqrt(T)`` beyond the farther of
            ln S0 / ln K on each side (default 6).
    """

    def __init__(self, S0, K, sigma, T, n_s=401, beta=6.0):
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be positive")
        if sigma <= 0 or T <= 0:
            raise ValueError("sigma and T must be positive")
        if n_s < 5:
            raise ValueError("n_s must be at least 5")
        self.S0 = float(S0)
        self.K = float(K)
        self.sigma = float(sigma)
        self.T = float(T)
        self.beta = float(beta)
        self.n = n_s if n_s % 2 == 1 else n_s + 1

        x0 = np.log(self.S0)
        xk = np.log(self.K)
        s = self.sigma * np.sqrt(self.T)  # log-price diffusion scale
        left = (x0 - min(x0, xk)) + self.beta * s
        right = (max(x0, xk) - x0) + self.beta * s
        halfwidth = max(left, right)

        self.x0 = x0
        self.i0 = (self.n - 1) // 2
        self.h = 2.0 * halfwidth / (self.n - 1)
        self.x = x0 + (np.arange(self.n) - self.i0) * self.h
        self.S = np.exp(self.x)
        self.xmin = float(self.x[0])
        self.xmax = float(self.x[-1])

    def margin(self):
        """Smallest distance (in x) from ln S0 / ln K to the domain ends."""
        xk = np.log(self.K)
        inner = min(self.x0, xk)
        outer = max(self.x0, xk)
        return float(min(outer - self.xmin, self.xmax - inner))

    def initial_values(self, kind="call"):
        """Payoff at every node, the solution at tau = 0."""
        return payoff(self.S, self.K, kind)
