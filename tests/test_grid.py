"""LogGrid construction: node placement, domain width, payoff initial data.

The grid is centered on x0 = ln S0 (S0 is always a grid node) and extends
at least ``beta * sigma * sqrt(T)`` beyond both ln S0 and the strike ln K
in x, so the far-field truncation error is a negligible Gaussian tail.
"""

import math

import numpy as np
import pytest

from option_pricer.grid import LogGrid
from option_pricer.payoff import payoff

BETA = 6.0


def _grid(S0=100.0, K=100.0, n_s=401):
    return LogGrid(S0, K, 0.2, 1.0, n_s=n_s, beta=BETA)


def test_s0_is_center_node():
    g = _grid(S0=105.0, K=100.0)
    assert g.n == 401
    assert g.i0 == 200
    assert math.isclose(g.x[g.i0], math.log(105.0))
    assert math.isclose(g.S[g.i0], 105.0)


def test_odd_node_count_enforced():
    assert _grid(n_s=400).n == 401  # even request rounded up to odd


def test_uniform_spacing_and_ordering():
    g = _grid()
    np.testing.assert_allclose(np.diff(g.x), g.h, rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(g.S) > 0.0)
    assert g.h > 0.0


def test_margin_from_spot_and_strike():
    """Both ln S0 and ln K must sit >= beta*sigma*sqrt(T) from the ends."""
    s = 0.2 * math.sqrt(1.0)
    for S0, K in ((100.0, 100.0), (105.0, 100.0), (100.0, 300.0),
                  (70.0, 100.0)):
        g = _grid(S0=S0, K=K)
        xk = math.log(K)
        assert g.margin() >= BETA * s - 1e-12
        assert min(xk - g.xmin, g.xmax - xk) >= BETA * s - 1e-12
        assert min(g.x0 - g.xmin, g.xmax - g.x0) >= BETA * s - 1e-12


def test_refining_halves_h():
    """Doubling n_s must halve the step h (the domain width is fixed)."""
    h1 = _grid(n_s=201).h
    h2 = _grid(n_s=401).h
    assert math.isclose(h1 / h2, 2.0, rel_tol=1e-12)


def test_initial_values_are_payoff():
    g = _grid(S0=105.0, K=100.0)
    np.testing.assert_array_equal(g.initial_values("call"),
                                  payoff(g.S, 100.0, "call"))
    np.testing.assert_array_equal(g.initial_values("put"),
                                  payoff(g.S, 100.0, "put"))


@pytest.mark.parametrize("S0,K,sigma,T", [
    (0.0, 100.0, 0.2, 1.0), (-5.0, 100.0, 0.2, 1.0),
    (100.0, 0.0, 0.2, 1.0), (100.0, 100.0, -0.2, 1.0),
    (100.0, 100.0, 0.2, 0.0), (100.0, 100.0, 0.2, -1.0),
])
def test_invalid_parameters_raise(S0, K, sigma, T):
    with pytest.raises(ValueError):
        LogGrid(S0, K, sigma, T, n_s=401, beta=BETA)


def test_too_few_nodes_raise():
    with pytest.raises(ValueError):
        LogGrid(100.0, 100.0, 0.2, 1.0, n_s=3)
