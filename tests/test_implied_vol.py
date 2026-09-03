"""Implied volatility solver: uniqueness, recovery and no-arbitrage bounds.

The Black-Scholes price is strictly increasing in sigma (vega > 0), so a
price produced at a known sigma must be inverted back to that sigma to
machine precision, and the same sigma must be recovered from the call and
from the put (put-call parity implies equal implied volatilities).  Prices
outside the no-arbitrage bounds are rejected.
"""

import math

import pytest

from option_pricer.analytic import bs_price
from option_pricer.implied_vol import implied_volatility, price_bounds


def _cases():
    for S, K, T, r in ((100.0, 100.0, 1.0, 0.05),
                       (105.0, 100.0, 1.0, 0.05),
                       (80.0, 100.0, 0.5, 0.03),
                       (120.0, 100.0, 2.0, 0.07)):
        for kind in ("call", "put"):
            yield S, K, T, r, kind


@pytest.mark.parametrize("S,K,T,r,kind", list(_cases()))
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6])
def test_recovery(S, K, T, r, kind, sigma):
    """A price made at sigma must invert back to sigma exactly.

    Tolerance 1e-7: for well-conditioned (large vega) cases the recovery is
    ~1e-14, but a deep in-the-money option at very low volatility has vega
    ~1e-7, so the price carries sigma to only ~1e-8 there.
    """
    price = bs_price(S, K, T, r, sigma, kind)
    assert abs(implied_volatility(price, S, K, T, r, kind) - sigma) < 1e-7


def test_call_and_put_give_same_implied_vol():
    """Put-call parity: equal sigma means call and put invert identically."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma = 0.25
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    assert abs(implied_volatility(c, S, K, T, r, "call")
               - implied_volatility(p, S, K, T, r, "put")) < 1e-9


def test_implied_vol_increases_with_price():
    """The inversion must be monotone in the observed price."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    low = bs_price(S, K, T, r, 0.15, "call")
    high = bs_price(S, K, T, r, 0.35, "call")
    assert (implied_volatility(high, S, K, T, r, "call")
            > implied_volatility(low, S, K, T, r, "call"))


def test_bounds_reject_arbitrage_prices():
    """Prices at or beyond the no-arbitrage bounds have no implied vol."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    df = K * math.exp(-r * T)
    for bad, kind in ((-1.0, "call"),        # negative
                      (S + 1.0, "call"),     # above the upper bound S
                      (S - df - 1e-12, "call"),   # at/below the lower bound
                      (0.0, "put"),
                      (df + 1e-12, "put")):
        with pytest.raises(ValueError):
            implied_volatility(bad, S, K, T, r, kind)


def test_bounds_shapes():
    """Sanity on the interval returned by price_bounds."""
    lo, hi = price_bounds(100.0, 100.0, 1.0, 0.05, "call")
    assert 0.0 < lo < hi  # S - K e^{-rT} > 0 for this ATM-ish case
    lo_p, hi_p = price_bounds(120.0, 100.0, 1.0, 0.05, "put")
    assert lo_p == 0.0 and hi_p > 0.0
