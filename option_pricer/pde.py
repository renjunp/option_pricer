"""Semi-discrete Black-Scholes operator and far-field boundary data.

With ``x = ln S`` and ``tau = T - t`` the Black-Scholes PDE

    V_t + 1/2 sigma^2 S^2 V_SS + r S V_S - r V = 0

becomes the constant-coefficient forward-parabolic equation

    V_tau = 1/2 sigma^2 V_xx + (r - 1/2 sigma^2) V_x - r V.

Central differences on a uniform grid of step ``h`` replace the derivatives,
giving the constant tridiagonal operator ``A`` (per interior node):

    A V_i = a_up V_{i+1} + a_diag V_i + a_low V_{i-1},
    a_low  = sigma^2/(2 h^2) - mu/(2 h),     mu = r - sigma^2/2,
    a_diag = -sigma^2/h^2 - r,
    a_up   = sigma^2/(2 h^2) + mu/(2 h).

The theta-scheme step on the interior unknowns is

    (I - dt theta A) V^{n+1} = (I + dt (1-theta) A) V^n + boundary terms,

where the boundary nodes are fixed by the far-field asymptotics and their
(possibly time-dependent) values are moved to the right-hand side each step.
"""

import numpy as np
from scipy.sparse import diags


def far_field_boundary(kind, grid, r, tau):
    """Dirichlet far-field data (V_left, V_right) at forward time ``tau``.

    Far from the strike the option approaches its linear asymptote:
        call: V -> 0 as S -> 0,   V -> S - K e^{-r tau} as S -> inf,
        put:  V -> K e^{-r tau} as S -> 0,   V -> 0 as S -> inf.
    """
    df = np.exp(-r * tau)
    if kind == "call":
        return 0.0, float(np.exp(grid.xmax) - grid.K * df)
    if kind == "put":
        return float(grid.K * df), 0.0
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


class ThetaOperator:
    """Theta-scheme step matrices on the interior nodes.

    Attributes:
        L, R: sparse (m, m) matrices with L V^{n+1} = R V^n + boundary rhs,
        a_low, a_up: stencil coefficients coupling the first/last interior
            rows to the left/right boundary node values,
        m: number of interior nodes (n - 2).
    """

    def __init__(self, grid, sigma, r, dt, theta=0.5):
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        h = grid.h
        mu = r - 0.5 * sigma ** 2
        a_low = sigma ** 2 / (2.0 * h * h) - mu / (2.0 * h)
        a_diag = -sigma ** 2 / (h * h) - r
        a_up = sigma ** 2 / (2.0 * h * h) + mu / (2.0 * h)
        self.a_low, self.a_up = a_low, a_up

        m = grid.n - 2
        # L = I - dt theta A,  R = I + dt (1-theta) A  on interior nodes.
        L = diags(
            [-dt * theta * a_low, 1.0 - dt * theta * a_diag, -dt * theta * a_up],
            [-1, 0, 1], shape=(m, m), format="csr",
        )
        R = diags(
            [dt * (1.0 - theta) * a_low, 1.0 + dt * (1.0 - theta) * a_diag,
             dt * (1.0 - theta) * a_up],
            [-1, 0, 1], shape=(m, m), format="csr",
        )
        self.L, self.R, self.m = L, R, m
