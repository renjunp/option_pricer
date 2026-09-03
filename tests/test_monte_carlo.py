"""Monte Carlo with antithetic variates against the Black-Scholes benchmark.

With p antithetic pairs the estimator averages h(Z) = (f(Z) + f(-Z)) / 2
of the discounted payoffs; the acceptance criterion is a relative error
below 5e-3 at n_paths >= 1e5.  Tests use n_paths = 1e6 with a fixed seed,
where the realized error stays a small multiple of the reported standard
error, and check that the antithetic variance reduction factor exceeds 1.
"""

from option_pricer.analytic import bs_price
from option_pricer.monte_carlo import mc_price
from option_pricer.postprocess import relative_error

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
N_PATHS = 1_000_000
SEED = 0


def test_reproducible_with_seed():
    a = mc_price(S0, K, T, R, SIGMA, "call", n_paths=N_PATHS, seed=SEED)
    b = mc_price(S0, K, T, R, SIGMA, "call", n_paths=N_PATHS, seed=SEED)
    assert a.price == b.price
    assert a.crude_price == b.crude_price
    assert a.se == b.se
    assert a.variance_ratio == b.variance_ratio


def test_different_seeds_differ():
    a = mc_price(S0, K, T, R, SIGMA, "call", n_paths=N_PATHS, seed=1)
    b = mc_price(S0, K, T, R, SIGMA, "call", n_paths=N_PATHS, seed=2)
    assert a.price != b.price


def test_call_within_5e3_relative():
    m = mc_price(S0, K, T, R, SIGMA, "call", n_paths=N_PATHS, seed=SEED)
    exact = bs_price(S0, K, T, R, SIGMA, "call")
    assert relative_error(m.price, exact) < 5e-3


def test_put_within_5e3_relative():
    m = mc_price(S0, K, T, R, SIGMA, "put", n_paths=N_PATHS, seed=SEED)
    exact = bs_price(S0, K, T, R, SIGMA, "put")
    assert relative_error(m.price, exact) < 5e-3


def test_error_consistent_with_reported_se():
    """The realized error should be a few standard errors at most."""
    exact = bs_price(S0, K, T, R, SIGMA, "call")
    m = mc_price(S0, K, T, R, SIGMA, "call", n_paths=N_PATHS, seed=SEED)
    assert abs(m.price - exact) < 4.0 * m.se


def test_antithetic_reduces_variance():
    """The antithetic estimator variance must be below the crude one."""
    for kind in ("call", "put"):
        m = mc_price(S0, K, T, R, SIGMA, kind, n_paths=N_PATHS, seed=SEED)
        assert m.variance_ratio > 1.0
        assert m.crude_se > m.se
