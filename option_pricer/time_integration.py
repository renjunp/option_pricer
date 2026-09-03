"""Time integration of the semi-discrete Black-Scholes PDE.

Starting from the payoff at ``tau = 0``, the theta-scheme (backward Euler,
``theta = 1``, first order; Crank-Nicolson, ``theta = 1/2``, second order;
both A-stable, so no CFL restriction) marches forward in ``tau = T - t`` to
the pricing time ``tau = T``.  The step matrix is factorized once with
``splu`` and each step solves a tridiagonal system on the interior nodes,
with the far-field Dirichlet data substituted into the right-hand side.
"""

import numpy as np
from scipy.sparse.linalg import splu

from .grid import LogGrid
from .pde import ThetaOperator, far_field_boundary


class FDResult:
    """Output of :func:`solve_fd`.

    Attributes:
        price: option value at the quoted spot S0 (center grid node),
        V: final price vector on the grid at tau = T,
        grid: the LogGrid used,
        scheme, dt, n_steps: scheme label and time discretization,
        max_residual: largest per-step residual of the interior system,
        U: full solution history (n_steps+1, n) if ``history`` was True.
    """

    def __init__(self, price, V, grid, scheme, dt, n_steps, max_residual, U=None):
        self.price = price
        self.V = V
        self.grid = grid
        self.scheme = scheme
        self.dt = dt
        self.n_steps = n_steps
        self.max_residual = max_residual
        self.U = U


def solve_fd(S0, K, T, r, sigma, kind="call", n_s=401, n_t=400, scheme="cn",
             beta=6.0, history=False):
    """Price a European call/put by finite differences on ``x = ln S``.

    scheme: "be" (backward Euler, 1st order in dt) or "cn"
    (Crank-Nicolson, 2nd order in dt).  Returns an FDResult; the price is
    read at the center node ``ln S0`` after ``n_t`` steps to ``tau = T``.
    """
    if scheme not in ("be", "cn"):
        raise ValueError(f"unsupported scheme: {scheme!r} (use 'be' or 'cn')")
    theta = 1.0 if scheme == "be" else 0.5
    if n_t < 1:
        raise ValueError("n_t must be at least 1")

    grid = LogGrid(S0, K, sigma, T, n_s=n_s, beta=beta)
    dt = T / n_t
    ops = ThetaOperator(grid, sigma, r, dt, theta)
    lu = splu(ops.L.tocsc())

    v = grid.initial_values(kind)
    max_residual = 0.0
    if history:
        U = [v.copy()]
    else:
        U = None

    for step in range(n_t):
        tau_old = step * dt
        tau_new = tau_old + dt
        bl_old, br_old = far_field_boundary(kind, grid, r, tau_old)
        bl_new, br_new = far_field_boundary(kind, grid, r, tau_new)

        rhs = ops.R @ v[1:-1]
        # Interior rows adjacent to a boundary node pick up the Dirichlet
        # value through the a_low / a_up stencil coefficients.
        rhs[0] += dt * (theta * ops.a_low * bl_new
                        + (1.0 - theta) * ops.a_low * bl_old)
        rhs[-1] += dt * (theta * ops.a_up * br_new
                         + (1.0 - theta) * ops.a_up * br_old)

        x = lu.solve(rhs)
        max_residual = max(max_residual, float(np.max(np.abs(ops.L @ x - rhs))))

        v = np.empty(grid.n)
        v[0] = bl_new
        v[-1] = br_new
        v[1:-1] = x
        if history:
            U.append(v.copy())

    return FDResult(
        price=float(v[grid.i0]),
        V=v,
        grid=grid,
        scheme=scheme,
        dt=dt,
        n_steps=n_t,
        max_residual=max_residual,
        U=np.array(U) if U is not None else None,
    )


def grid_greeks(res, r, sigma, kind="call", eps_sigma=1e-3):
    """(delta, gamma, vega) of a finite-difference solution at S0.

    delta and gamma come from central differences of the final vector in
    ``x``, converted with ``d/dS = (1/S) d/dx`` and
    ``d^2/dS^2 = (1/S^2)(d^2/dx^2 - d/dx)``.  vega needs a second solve at
    ``sigma +/- eps_sigma``; both uses of the FD solver share the same grid
    size and step count, so the difference is a pure volatility bump.
    """
    g = res.grid
    i0 = g.i0
    dx = (res.V[i0 + 1] - res.V[i0 - 1]) / (2.0 * g.h)
    d2x = (res.V[i0 + 1] - 2.0 * res.V[i0] + res.V[i0 - 1]) / (g.h * g.h)
    delta = dx / g.S0
    gamma = (d2x - dx) / (g.S0 * g.S0)

    eps = eps_sigma * sigma
    vp = solve_fd(g.S0, g.K, g.T, r, sigma + eps, kind,
                  n_s=g.n, n_t=res.n_steps, scheme=res.scheme, beta=g.beta)
    vm = solve_fd(g.S0, g.K, g.T, r, sigma - eps, kind,
                  n_s=g.n, n_t=res.n_steps, scheme=res.scheme, beta=g.beta)
    vega = (vp.price - vm.price) / (2.0 * eps)
    return delta, gamma, vega
