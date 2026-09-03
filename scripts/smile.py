#!/usr/bin/env python
"""Volatility smile demo for the option pricer.

Synthetic "market" prices are generated from a parameterized smile: the
market implied volatility is the quadratic-in-log-moneyness function

    sigma_mkt(K) = sigma0 + beta * ln(K/S0) + gamma * ln(K/S0)^2,

plugged into the Black-Scholes closed form.  Each strike's price is then
inverted with option_pricer.implied_volatility; the recovered curve lies
on top of sigma_mkt to ~1e-13 (1e-8 at the far wings), demonstrating the
inverter.  A single constant volatility can never reproduce such prices
across all strikes -- that is the smile.

Output: aligned table on stdout, results/smile_iv.csv, results/smile_iv.png.

Usage (from the repository root):

    .venv/bin/python scripts/smile.py
"""

import csv
import pathlib

import numpy as np

from option_pricer.analytic import bs_price
from option_pricer.implied_vol import implied_volatility
from option_pricer.visualize import plot_order  # noqa: F401  (forces Agg)
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

S0, R, T = 100.0, 0.05, 1.0
SIGMA0, BETA, GAMMA = 0.20, -0.04, 0.35   # smile parameters
N_STRIKES = 31
KIND = "call"


def sigma_market(K):
    """Synthetic market implied volatility: smile in log-moneyness."""
    x = np.log(K / S0)
    return SIGMA0 + BETA * x + GAMMA * x * x


def main():
    OUT.mkdir(exist_ok=True)
    strikes = S0 * np.linspace(0.55, 1.65, N_STRIKES)
    rows = []
    print(f"implied volatility smile, {KIND}s, S0 = {S0:g}, T = {T:g}, "
          f"r = {R:g}")
    print(f"smile model: sigma(K) = {SIGMA0} {BETA:+g}*ln(K/S0) "
          f"{GAMMA:+g}*ln(K/S0)^2")
    print(f"{'K':>8} {'K/S0':>8} {'sigma_mkt':>11} {'sigma_iv':>11} "
          f"{'|diff|':>9}")
    for K in strikes:
        vol = float(sigma_market(K))
        price = bs_price(S0, K, T, R, vol, KIND)          # "market" price
        iv = implied_volatility(price, S0, K, T, R, KIND)  # inverted
        rows.append((float(K), float(K / S0), vol, iv, abs(iv - vol)))
        print(f"{K:>8.2f} {K / S0:>8.3f} {vol:>11.6f} {iv:>11.6f} "
              f"{abs(iv - vol):>9.1e}")

    path = OUT / "smile_iv.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("K", "moneyness", "sigma_market", "sigma_implied",
                         "abs_error"))
        writer.writerows(rows)
    print(f"wrote {path}")

    moneyness = np.array([r[1] for r in rows])
    market = np.array([r[2] for r in rows])
    ivs = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(moneyness, ivs, "o-", ms=4, label="implied vol (inverted)")
    ax.plot(moneyness, market, "--", color="gray",
            label="synthetic market vol")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("moneyness  K/S0")
    ax.set_ylabel("implied volatility")
    ax.set_title(f"Volatility smile ({KIND}, T = {T:g}, S0 = {S0:g})")
    ax.legend()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * S0, ax.get_xlim()[1] * S0)
    ax2.set_xlabel("strike K")
    fig.tight_layout()
    fig_path = OUT / "smile_iv.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
