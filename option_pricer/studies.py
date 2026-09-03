"""Convergence-study drivers shared by the pytest suite and
``scripts/convergence.py`` so the two cannot drift apart.

Temporal orders are measured against a fine-dt reference solution of the
*same* scheme on the same grid, which isolates the time-discretization
error from the (identical) spatial truncation.  The pricing point for the
temporal studies lies a little off the strike (S0 = 105, K = 100) so the
payoff kink at ln K does not contaminate the measured time rates.

The spatial order is measured with the quoted spot on the strike
(S0 = K = 100): with the grid symmetric about the payoff kink the
pointwise error at S0 converges cleanly at second order, and taking
dt = h/4 keeps the Crank-Nicolson time error O(dt^2) = O(h^2/16) far
below the measured spatial error.  Errors are relative to the
Black-Scholes closed form.
"""

from .analytic import bs_price
from .grid import LogGrid
from .postprocess import relative_error
from .time_integration import solve_fd

# Shared study configuration ----------------------------------------------
TIME_NS = 401        # spatial nodes for the temporal studies
REF_STEPS = 2 ** 15  # dt of the temporal reference solution: T / REF_STEPS
STUDY_S0, STUDY_K = 105.0, 100.0
STUDY_R, STUDY_SIGMA, STUDY_T = 0.05, 0.2, 1.0

SPACE_NS = (101, 201, 401, 801)   # spatial studies refine this list
SPACE_S0, SPACE_K = 100.0, 100.0  # ATM: grid symmetric about the kink
SPACE_DT_FACTOR = 4.0             # dt = h / factor, keeps dt^2 << h^2


def time_study(scheme, S0=STUDY_S0, K=STUDY_K, T=STUDY_T, r=STUDY_R,
               sigma=STUDY_SIGMA, kind="call", n_s=TIME_NS, steps=None):
    """Price error vs dt for a fixed grid.

    steps: numbers of time steps (halved-dt sequence).  Defaults give three
    consecutive order measurements for each scheme:
        "be": (40, 80, 160, 320),   "cn": (80, 160, 320, 640).
    Returns a dict with steps, dt = T/steps, err_bs (vs Black-Scholes) and
    err_time (vs the fine-dt reference, the order that must be measured).
    """
    if steps is None:
        steps = (40, 80, 160, 320) if scheme == "be" else (80, 160, 320, 640)
    exact = bs_price(S0, K, T, r, sigma, kind)
    ref = solve_fd(S0, K, T, r, sigma, kind, n_s=n_s, n_t=REF_STEPS,
                   scheme=scheme).price
    result = {"steps": [], "dt": [], "err_bs": [], "err_time": []}
    for n_steps in steps:
        price = solve_fd(S0, K, T, r, sigma, kind, n_s=n_s, n_t=n_steps,
                         scheme=scheme).price
        result["steps"].append(n_steps)
        result["dt"].append(T / n_steps)
        result["err_bs"].append(relative_error(price, exact))
        result["err_time"].append(abs(price - ref))
    return result


def space_study(n_s_list=SPACE_NS, S0=SPACE_S0, K=SPACE_K, T=STUDY_T,
                r=STUDY_R, sigma=STUDY_SIGMA, kind="call", scheme="cn",
                dt_factor=SPACE_DT_FACTOR):
    """Price error vs grid spacing h (Crank-Nicolson, dt = h / dt_factor).

    Refining n_s doubles the nodes and halves h; dt follows h so the
    Crank-Nicolson time error stays O(h^2 / dt_factor^2) and never shows
    up in the measured spatial rate.  Returns a dict with n_s, h, err_bs
    and the exact Black-Scholes reference used.
    """
    exact = bs_price(S0, K, T, r, sigma, kind)
    result = {"n_s": [], "h": [], "err_bs": [], "exact": exact}
    for n_s in n_s_list:
        grid = LogGrid(S0, K, sigma, T, n_s=n_s)
        n_t = max(1, int(round(T / (grid.h / dt_factor))))
        res = solve_fd(S0, K, T, r, sigma, kind, n_s=n_s, n_t=n_t, scheme=scheme)
        result["n_s"].append(n_s)
        result["h"].append(grid.h)
        result["err_bs"].append(relative_error(res.price, exact))
    return result
