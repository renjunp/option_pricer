"""Finite-difference pricing against the Black-Scholes benchmark.

The acceptance configuration (S = K = 100, T = 1, r = 5%, sigma = 20%) is
priced with the dense default grid (n_s = 601 nodes, n_t = 500 steps):
Crank-Nicolson must be within 1e-4 relative of the closed form, backward
Euler within 1e-3 (first order in dt by design).  Because both schemes
discretize the same linear operator, the FD prices also satisfy
put-call parity to the truncation error.
"""

import math

import pytest

from option_pricer.analytic import bs_delta, bs_gamma, bs_price, bs_vega
from option_pricer.postprocess import relative_error
from option_pricer.time_integration import grid_greeks, solve_fd

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
NS, NT = 601, 500  # dense default grid


def _solve(scheme, kind="call"):
    return solve_fd(S0, K, T, R, SIGMA, kind, n_s=NS, n_t=NT, scheme=scheme)


def test_cn_atm_within_1e4():
    """Crank-Nicolson on the dense grid: relative error < 1e-4."""
    res = _solve("cn")
    assert relative_error(res.price, bs_price(S0, K, T, R, SIGMA, "call")) < 1e-4


def test_be_within_1e3():
    """Backward Euler is first order in dt: relax the bound to 1e-3."""
    res = _solve("be")
    assert relative_error(res.price, bs_price(S0, K, T, R, SIGMA, "call")) < 1e-3


def test_cn_put_within_1e3():
    res = _solve("cn", kind="put")
    assert relative_error(res.price, bs_price(S0, K, T, R, SIGMA, "put")) < 1e-3


def test_step_residuals_machine_precision():
    """The interior theta-system must solve to round-off each step."""
    for scheme in ("be", "cn"):
        assert _solve(scheme).max_residual < 1e-9


def test_fd_put_call_parity():
    """V_call - V_put = S0 - K e^{-rT} holds to the truncation error."""
    vc = _solve("cn", kind="call").price
    vp = _solve("cn", kind="put").price
    assert abs((vc - vp) - (S0 - K * math.exp(-R * T))) < 1e-4


def test_fd_greeks_match_black_scholes():
    """Grid-difference Greeks of the CN solution vs the closed form."""
    res = _solve("cn")
    delta, gamma, vega = grid_greeks(res, R, SIGMA, "call")
    assert abs(delta - bs_delta(S0, K, T, R, SIGMA, "call")) < 1e-3
    assert abs(gamma - bs_gamma(S0, K, T, R, SIGMA, "call")) < 1e-3
    assert abs(vega - bs_vega(S0, K, T, R, SIGMA, "call")) < 0.05


def test_invalid_scheme_raises():
    with pytest.raises(ValueError):
        solve_fd(S0, K, T, R, SIGMA, "call", n_s=NS, n_t=NT, scheme="explicit")
