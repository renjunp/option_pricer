"""Monte Carlo pricing with antithetic variance reduction.

The geometric Brownian motion is sampled exactly at maturity,

    S_T(Z) = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),

so no path discretization is needed.  With ``p`` antithetic pairs the
estimator averages

    h(Z) = (f(Z) + f(-Z)) / 2,   f = e^{-rT} payoff(S_T(Z)),

which cancels the odd part of the noise and lowers the variance of the
price estimate:

    V_hat = mean(h),   SE = std(h) / sqrt(p).

A crude (independent-sample) estimate over the same ``2p`` discounted
payoffs is kept alongside, and the variance-reduction factor is reported as
the ratio of the two estimator variances, ``var_crude / (2 var_h)``.
"""

import numpy as np

from .analytic import bs_greeks
from .payoff import payoff


class MCResult:
    """Output of :func:`mc_price`.

    Attributes:
        price: antithetic estimate of the discounted option price,
        se: standard error of ``price`` over the p pairs,
        n_pairs: number of antithetic pairs (floor(n_paths / 2)),
        crude_price, crude_se: plain Monte Carlo over the same 2p payoffs,
        variance_ratio: var_crude / (2 var_h), factor by which the antithetic
            estimator variance is smaller,
        greeks: (delta, gamma, vega) Black-Scholes closed form (reference
            values; used because finite-difference Greeks of a noisy Monte
            Carlo estimate are themselves noisy),
        seed: RNG seed used.
    """

    def __init__(self, price, se, n_pairs, crude_price, crude_se,
                 variance_ratio, greeks, seed):
        self.price = price
        self.se = se
        self.n_pairs = n_pairs
        self.crude_price = crude_price
        self.crude_se = crude_se
        self.variance_ratio = variance_ratio
        self.greeks = greeks
        self.seed = seed


def mc_price(S0, K, T, r, sigma, kind="call", n_paths=1_000_000, seed=0):
    """Antithetic Monte Carlo price of a European call/put at S0."""
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    n_pairs = n_paths // 2
    if n_pairs < 1:
        raise ValueError("n_paths must be at least 2")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_pairs)
    mu = (r - 0.5 * sigma ** 2) * T
    spread = sigma * np.sqrt(T)
    df = np.exp(-r * T)

    s_up = S0 * np.exp(mu + spread * z)
    s_dn = S0 * np.exp(mu - spread * z)
    f_up = df * payoff(s_up, K, kind)
    f_dn = df * payoff(s_dn, K, kind)

    h = 0.5 * (f_up + f_dn)  # antithetic pair average
    price = float(np.mean(h))
    var_h = float(np.var(h, ddof=1))
    se = float(np.sqrt(var_h / n_pairs))

    crude = np.concatenate([f_up, f_dn])
    crude_price = float(np.mean(crude))
    var_crude = float(np.var(crude, ddof=1))
    crude_se = float(np.sqrt(var_crude / (2.0 * n_pairs)))
    variance_ratio = var_crude / (2.0 * var_h) if var_h > 0.0 else float("inf")

    greeks = bs_greeks(S0, K, T, r, sigma, kind)
    return MCResult(price, se, n_pairs, crude_price, crude_se,
                    variance_ratio, greeks, seed)
