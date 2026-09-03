"""Implied volatility: invert the Black-Scholes formula for sigma.

The Black-Scholes price is strictly increasing in volatility (vega =
S phi(d1) sqrt(T) > 0 for sigma > 0), so the implied volatility of a
market price is the unique root of

    bs_price(sigma) - price = 0,

found with Brent's method (scipy.optimize.brentq).  A price outside the
no-arbitrage bounds can never be matched by any positive sigma and raises
a ValueError:

    call:  max(S - K e^{-rT}, 0) < price < S
    put:   max(K e^{-rT} - S, 0) < price < K e^{-rT}
"""

import math

from scipy.optimize import brentq

from .analytic import bs_price


def price_bounds(S, K, T, r, kind="call"):
    """No-arbitrage interval (lower, upper) that a price must lie in."""
    df = K * math.exp(-r * T)  # discounted strike
    if kind == "call":
        return max(S - df, 0.0), S
    if kind == "put":
        return max(df - S, 0.0), df
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def implied_volatility(price, S, K, T, r, kind="call", lo=1e-4, hi=2.0,
                       xtol=1e-12):
    """Black-Scholes implied volatility of a market price.

    price: observed option price at spot S, strike K, time to maturity T,
    rate r; kind: "call" or "put".  The price must lie strictly inside the
    no-arbitrage bounds from :func:`price_bounds`, otherwise no volatility
    reproduces it and a ValueError is raised.
    """
    lower, upper = price_bounds(S, K, T, r, kind)
    if price <= lower or price >= upper:
        raise ValueError(
            f"{kind} price {price:.6g} outside the no-arbitrage range "
            f"({lower:.6g}, {upper:.6g}) for S={S:g}, K={K:g}, T={T:g}, r={r:g}"
        )
    if lo >= hi:
        raise ValueError("lo must be smaller than hi")
    f = lambda sigma: bs_price(S, K, T, r, sigma, kind) - price
    return float(brentq(f, lo, hi, xtol=xtol))
