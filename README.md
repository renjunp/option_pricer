# option_pricer

European call/put pricing in Python: Black-Scholes closed form, finite
differences for the Black-Scholes PDE, and Monte Carlo with antithetic
variates, all cross-verified against each other.

## Methods

All formulas are for a European option with spot `S`, strike `K`, time to
maturity `T`, risk-free rate `r`, volatility `σ` (no dividends).

**1. Black-Scholes closed form** (benchmark truth):

```
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)      d2 = d1 − σ√T

call = S·N(d1) − K·e^{−rT}·N(d2)
put  = K·e^{−rT}·N(−d2) − S·N(−d1)
```

**2. Finite differences** on x = ln S with τ = T − t, θ-scheme in time
(θ = 1 implicit Euler → order 1 in dt; θ = 1/2 Crank-Nicolson → order 2):

```
V_τ = ½σ² V_xx + (r − ½σ²) V_x − r V          V(x,0) = payoff(e^x)

(I − dt·θ·A) V^{n+1} = (I + dt·(1−θ)·A) V^n + boundary terms
```

**3. Monte Carlo** with antithetic pairs `h(Z) = (f(Z) + f(−Z))/2`,
`f = e^{−rT}·payoff(S_T)`, `S_T = S0·exp((r−σ²/2)T + σ√T·Z)`:

```
V̂ = mean(h)     SE = std(h)/√p (p pairs)     ratio = Var_crude/(2·Var_h)
```

## Verification

Relative error is `|approx − exact|/exact` for positive prices.  Smoke
case S = K = 100, T = 1, r = 5%, σ = 20% (ATM call):

| method | price | rel err vs Black-Scholes |
|---|---|---|
| Black-Scholes (reference) | 10.4505835722 | — |
| Crank-Nicolson, n_s = 601, n_t = 500 | 10.450224 | 3.4e-5 (< 1e-4) |
| backward Euler, n_s = 601, n_t = 500 | 10.448124 | 2.4e-4 |
| Monte Carlo, 10⁶ paths, antithetic | 10.464331 ± 0.010412 | 1.3e-3 (< 5e-3) |

Monte Carlo put: 5.582566 ± 0.006628 (rel err 1.6e-3).  Antithetic
variance-reduction ratio measured ≈ 2.0 (call) / 1.7 (put).

Convergence orders (`scripts/convergence.py` reproduces these tables,
CSV and the figure `results/convergence_orders.png`):

| study | measured order | expectation |
|---|---|---|
| time, implicit Euler (dt → 0) | 1.000, 1.003, 1.007 | 1 |
| time, Crank-Nicolson (dt → 0) | 2.000, 2.000, 2.001 | 2 |
| space, x-grid (h → 0, CN) | 2.003, 2.001, 2.000 | 2 |

Temporal orders are measured against a fine-dt reference of the same
scheme on the same grid (isolating the time error); see DEVELOPMENT.md.

## Usage

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --method cn
.venv/bin/python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --method analytic
.venv/bin/python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method be
.venv/bin/python pricer.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --method mc --n-paths 1000000 --seed 0

.venv/bin/python scripts/convergence.py        # tables + results/*.csv + PNG
.venv/bin/python scripts/smile.py               # volatility smile demo -> results/smile_iv.*
```

Every CLI call prints the price and delta/gamma/vega consistent with the
method: closed form for `analytic`, grid differences (+ σ-bump for vega)
for `be`/`cn`, and the Black-Scholes reference Greeks for `mc` (the CLI
notes this, since differencing a noisy Monte Carlo price is noisy).

## Volatility smile demo

`scripts/smile.py` builds synthetic "market" prices from a parameterized
smile -- the market implied volatility is quadratic in log-moneyness,
`σ_mkt(K) = 0.20 − 0.04·ln(K/S0) + 0.35·ln(K/S0)²` (K/S0 from 0.55 to
1.65) -- prices each strike with the Black-Scholes closed form, then
inverts the price back to an implied volatility with
`option_pricer.implied_volatility` (Brent's method on the monotone
`bs_price(σ) − price`).  The recovered curve reproduces `σ_mkt` to
~1e-16 at the money and ~1e-13 at the wings (worst-conditioned deep
in-the-money cases ~1e-8), which is the self-consistency check of the
inverter:

```
       K     K/S0   sigma_mkt   sigma_iv    |diff|
   55.00    0.550    0.349007    0.349007   1.1e-16
  100.00    1.000    0.200000    0.200000    ...
  165.00    1.650    0.267741    0.267741   8.4e-15
```

The left wing (low strikes) carries a higher implied volatility than the
right -- a classic smile/skew shape that no single constant volatility
can produce across strikes, which is exactly the effect being plotted.
Outputs: aligned table on stdout plus `results/smile_iv.csv` and
`results/smile_iv.png`.

## Tests

```
.venv/bin/python -m pytest
```

71 tests cover the closed form (smoke reference, parity, asymptotics,
Greeks), the payoff, the log grid, the finite-difference accuracy and
residuals, the three convergence orders, the Monte Carlo estimator
(determinism, error bound, variance reduction), and the implied-volatility
solver (recovery, monotonicity, no-arbitrage bounds).

## Project Structure

```
option_pricer/
├── pyproject.toml
├── pricer.py                 # CLI: price + Greeks
├── option_pricer/
│   ├── analytic.py           # Black-Scholes + Greeks (benchmark)
│   ├── payoff.py             # terminal payoffs
│   ├── grid.py               # LogGrid on x = ln S
│   ├── pde.py                # θ-scheme operator + far-field data
│   ├── time_integration.py   # BE / CN time marching, grid Greeks
│   ├── monte_carlo.py        # antithetic MC
│   ├── implied_vol.py        # invert BS prices to implied volatility
│   ├── postprocess.py        # errors and orders
│   ├── studies.py            # shared convergence drivers
│   └── visualize.py          # plots
├── scripts/convergence.py    # order studies → tables/CSV/PNG
├── scripts/smile.py          # volatility smile demo → CSV/PNG
├── tests/                    # 7 modules, 71 tests
├── results/                  # generated by scripts/convergence.py & smile.py
└── DEVELOPMENT.md            # math, implementation notes, measured data
```

## Not implemented

- [ ] American options, dividends, exotic payoffs
- [ ] Greeks beyond delta/gamma/vega
- [ ] Explicit time stepping / non-uniform grids
- [ ] Git history (delivered locally; `git init` and push left to the owner)
