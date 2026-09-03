"""Black-Scholes closed-form prices and Greeks for European options.

For a European call/put with spot ``S``, strike ``K``, time to maturity ``T``,
risk-free rate ``r`` and volatility ``sigma`` (no dividends),

    d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)

    call = S N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S N(-d1)

with N the standard normal CDF.  Greeks (same d1, phi the standard normal PDF):

    delta_call = N(d1),   delta_put = N(d1) - 1
    gamma = phi(d1) / (S sigma sqrt(T))
    vega  = S phi(d1) sqrt(T)

``vega`` is quoted per unit volatility (dC/dsigma); all formulas assume
scalar inputs and are used as the benchmark truth for the numerical methods.
"""

import numpy as np
from scipy.stats import norm


def _d1d2(S, K, T, r, sigma):
    s = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / s
    return d1, d1 - s


def bs_price(S, K, T, r, sigma, kind="call"):
    """Black-Scholes price of a European call/put (scalar inputs)."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    df = np.exp(-r * T)
    if kind == "call":
        return float(S * norm.cdf(d1) - K * df * norm.cdf(d2))
    if kind == "put":
        return float(K * df * norm.cdf(-d2) - S * norm.cdf(-d1))
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_delta(S, K, T, r, sigma, kind="call"):
    """Closed-form delta (d price / d spot)."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    if kind == "call":
        return float(norm.cdf(d1))
    if kind == "put":
        return float(norm.cdf(d1) - 1.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_gamma(S, K, T, r, sigma, kind="call"):
    """Closed-form gamma (d^2 price / d spot^2), same for call and put."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(S, K, T, r, sigma, kind="call"):
    """Closed-form vega (d price / d sigma), same for call and put."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * np.sqrt(T))


def bs_greeks(S, K, T, r, sigma, kind="call"):
    """Closed-form (delta, gamma, vega) as a tuple."""
    return (
        bs_delta(S, K, T, r, sigma, kind),
        bs_gamma(S, K, T, r, sigma, kind),
        bs_vega(S, K, T, r, sigma, kind),
    )


def parity_mismatch(call, put, S, K, T, r):
    """Absolute put-call parity residual ``call - put - (S - K e^{-rT})``."""
    return abs(call - put - (S - K * np.exp(-r * T)))
