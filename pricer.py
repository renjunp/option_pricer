"""CLI for the European option pricer.

Prices a European call/put by the Black-Scholes closed form, by finite
differences on x = ln S (backward Euler or Crank-Nicolson), or by Monte
Carlo with antithetic variates, and prints the price together with the
Greeks (delta/gamma/vega) consistent with the selected method.

Examples:
    python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --method cn
    python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --method analytic
    python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method be
    python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --method mc --n-paths 1000000 --seed 0
"""

import argparse

import numpy as np

from option_pricer.analytic import bs_greeks, bs_price
from option_pricer.monte_carlo import mc_price
from option_pricer.postprocess import relative_error
from option_pricer.time_integration import grid_greeks, solve_fd

METHOD_LABELS = {
    "analytic": "Black-Scholes closed form",
    "be": "finite difference, backward Euler",
    "cn": "finite difference, Crank-Nicolson",
    "mc": "Monte Carlo, antithetic variates",
}


def _validate(args, parser):
    if args.S <= 0 or args.K <= 0:
        parser.error("--S and --K must be positive")
    if args.T <= 0:
        parser.error("--T must be positive")
    if args.sigma <= 0:
        parser.error("--sigma must be positive")
    if args.method in ("be", "cn") and args.n_s < 5:
        parser.error("--n-s must be at least 5")
    if args.method in ("be", "cn") and args.n_t < 1:
        parser.error("--n-t must be at least 1")
    if args.method == "mc" and args.n_paths < 2:
        parser.error("--n-paths must be at least 2")
    if args.plot and args.method not in ("be", "cn"):
        parser.error("--plot is only supported with --method be or --method cn")


def _run_fd(args, scheme):
    res = solve_fd(args.S, args.K, args.T, args.r, args.sigma, args.kind,
                   n_s=args.n_s, n_t=args.n_t, scheme=scheme)
    delta, gamma, vega = grid_greeks(res, args.r, args.sigma, args.kind)
    exact = bs_price(args.S, args.K, args.T, args.r, args.sigma, args.kind)
    return {
        "label": METHOD_LABELS[scheme],
        "price": res.price,
        "extra": (f"grid n_s={res.grid.n}, n_t={res.n_steps}, "
                  f"dt={res.dt:.6g}"),
        "rel_err": relative_error(res.price, exact),
        "greeks": (delta, gamma, vega),
        "res": res,
    }


def _run_mc(args):
    m = mc_price(args.S, args.K, args.T, args.r, args.sigma, args.kind,
                 n_paths=args.n_paths, seed=args.seed)
    exact = bs_price(args.S, args.K, args.T, args.r, args.sigma, args.kind)
    return {
        "label": METHOD_LABELS["mc"],
        "price": m.price,
        "extra": (f"{m.n_pairs} antithetic pairs from "
                  f"{2 * m.n_pairs} payoffs, seed={m.seed}"),
        "se": m.se,
        "crude_se": m.crude_se,
        "variance_ratio": m.variance_ratio,
        "rel_err": relative_error(m.price, exact),
        "greeks": m.greeks,
        "mc": m,
    }


def _run_analytic(args):
    price = bs_price(args.S, args.K, args.T, args.r, args.sigma, args.kind)
    return {
        "label": METHOD_LABELS["analytic"],
        "price": price,
        "extra": "reference value",
        "rel_err": None,
        "greeks": bs_greeks(args.S, args.K, args.T, args.r, args.sigma, args.kind),
    }


def _print(result, args):
    print(f"kind: {args.kind}")
    print(f"params: S0 = {args.S:g}, K = {args.K:g}, T = {args.T:g}, "
          f"r = {args.r:g}, sigma = {args.sigma:g}")
    print(f"method: {result['label']} ({result['extra']})")
    delta, gamma, vega = result["greeks"]
    print(f"price = {result['price']:.8f}")
    if result.get("se") is not None:
        print(f"  +/- {result['se']:.6f} (1 sigma)")
        print(f"  crude SE (no variance reduction): "
              f"+/- {result['crude_se']:.6f}")
        print(f"  variance reduction vs crude: {result['variance_ratio']:.3f}x")
    if result["rel_err"] is not None:
        print(f"rel err vs Black-Scholes = {result['rel_err']:.3e}")
    print(f"delta = {delta:.6f}")
    print(f"gamma = {gamma:.6f}")
    print(f"vega  = {vega:.6f}")
    if result.get("mc") is not None:
        print("(MC Greeks are the Black-Scholes reference values: finite-")
        print(" difference Greeks of a noisy Monte Carlo price are noisy.)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Price a European option and report price + Greeks.")
    parser.add_argument("--S", type=float, default=100.0, help="spot price")
    parser.add_argument("--K", type=float, default=100.0, help="strike")
    parser.add_argument("--T", type=float, default=1.0, help="time to maturity")
    parser.add_argument("--r", type=float, default=0.05, help="risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.20, help="volatility")
    parser.add_argument("--kind", choices=("call", "put"), default="call")
    parser.add_argument("--method", choices=("analytic", "be", "cn", "mc"),
                        default="analytic")
    parser.add_argument("--n-s", type=int, default=601, dest="n_s",
                        help="grid nodes for finite difference (odd)")
    parser.add_argument("--n-t", type=int, default=500, dest="n_t",
                        help="time steps for finite difference")
    parser.add_argument("--n-paths", type=int, default=1_000_000, dest="n_paths",
                        help="total Monte Carlo paths (antithetic pairs = n/2)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for MC")
    parser.add_argument("--plot", metavar="PATH", default=None,
                        help="save a price-profile PNG (methods be/cn)")
    args = parser.parse_args(argv)
    _validate(args, parser)

    if args.method == "analytic":
        result = _run_analytic(args)
    elif args.method in ("be", "cn"):
        result = _run_fd(args, args.method)
    else:
        result = _run_mc(args)
    _print(result, args)

    if args.plot:
        from option_pricer.visualize import price_profile
        res = result["res"]
        S = res.grid.S
        V_bs = np.array(
            [bs_price(s, args.K, args.T, args.r, args.sigma, args.kind)
             for s in S])
        price_profile(S, res.V, V_bs, args.S, args.K, args.kind, args.plot)
        print(f"saved price profile -> {args.plot}")


if __name__ == "__main__":
    main()
