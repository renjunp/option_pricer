#!/usr/bin/env python
"""Convergence verification for the finite-difference option pricer.

Runs the three order studies implemented in option_pricer.studies (the same
drivers the pytest suite asserts on) and reports them as aligned tables on
stdout, as CSV files and as one log-log figure under ``results/``:

    backward Euler (time):  expected order 1 in dt
    Crank-Nicolson (time):  expected order 2 in dt
    space (x grid, CN):     expected order 2 in h

Usage (from the repository root):

    .venv/bin/python scripts/convergence.py
"""

import csv
import pathlib

import numpy as np

from option_pricer.postprocess import convergence_rates
from option_pricer.studies import space_study, time_study
from option_pricer.visualize import plot_order
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def _print_table(title, rows, headers, order_idx=None):
    """rows: list of tuples; order_idx: index of a column whose entry is
    computed between the previous and the current row (None -> blank)."""
    print(f"\n{title}")
    head = f"{headers[0]:>8} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12}"
    if order_idx is not None:
        head += f" {headers[4]:>8}"
    print(head)
    prev_err = None
    for row in rows:
        line = "".join(
            f"{row[i]:>8}" if i == 0 else f"{row[i]:>12.4e}"
            for i in range(4)
        )
        if order_idx is not None:
            err = row[order_idx]
            order = (np.log(prev_err / err) / np.log(2.0)
                     if prev_err is not None else float("nan"))
            line += f"{order:>8.3f}" if prev_err is not None else f"{'--':>8}"
            prev_err = err
        print(line)


def _time_rows(study):
    steps = study["steps"]
    return list(zip(steps, study["dt"], study["err_bs"], study["err_time"]))


def _run_time(title, fname, scheme):
    study = time_study(scheme)
    headers = ("n_steps", "dt", "err_vs_BS", "err_vs_ref", "order")
    rows = _time_rows(study)
    _print_table(title, rows, headers, order_idx=3)
    _write_csv(fname, headers[:4], rows)


def _run_space():
    study = space_study()
    rows = list(zip(study["n_s"], study["h"], study["err_bs"]))
    headers = ("n_s", "h", "err_vs_BS")
    print("\nSpace order (Crank-Nicolson on x = ln S, dt = h/4, "
          "ATM S0 = K = 100): expected order 2 in h")
    print(f"{headers[0]:>8} {headers[1]:>12} {headers[2]:>12} {'order':>8}")
    prev = None
    for n_s, h, err in rows:
        order = (np.log(prev[2] / err) / np.log(2.0)
                 if prev is not None else float("nan"))
        line = f"{n_s:>8} {h:>12.4e} {err:>12.4e}"
        line += f"{order:>8.3f}" if prev is not None else f"{'--':>8}"
        print(line)
        prev = (n_s, h, err)
    _write_csv("space", headers, rows)
    return rows


def _write_csv(name, headers, rows):
    path = OUT / f"convergence_{name}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"wrote {path}")


def _plot(space_rows, be_study, cn_study):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, study, scheme, order in (
        (axes[0], be_study, "be", 1),
        (axes[1], cn_study, "cn", 2),
    ):
        plot_order(ax, study["dt"], study["err_time"], order,
                   "dt (time step)", f"{scheme.upper()} time error")
    axes[0].set_title("Backward Euler: order 1 in dt")
    axes[1].set_title("Crank-Nicolson: order 2 in dt")

    ns = [r[0] for r in space_rows]
    hs = [r[1] for r in space_rows]
    errs = [r[2] for r in space_rows]
    plot_order(axes[2], hs, errs, 2, "h (grid spacing)", "space error")
    axes[2].set_title("Space (x = ln S): order 2 in h")

    fig.suptitle("Option pricer: measured convergence vs theoretical order")
    fig.tight_layout()
    path = OUT / "convergence_orders.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    be = time_study("be")
    cn = time_study("cn")
    _print_table("Time order, backward Euler (fixed n_s = 401): "
                 "expected order 1 in dt",
                 _time_rows(be), ("n_steps", "dt", "err_vs_BS", "err_vs_ref",
                                  "order"), order_idx=3)
    _write_csv("be_time", ("n_steps", "dt", "err_vs_BS", "err_vs_ref"),
               _time_rows(be))
    _print_table("Time order, Crank-Nicolson (fixed n_s = 401): "
                 "expected order 2 in dt",
                 _time_rows(cn), ("n_steps", "dt", "err_vs_BS", "err_vs_ref",
                                  "order"), order_idx=3)
    _write_csv("cn_time", ("n_steps", "dt", "err_vs_BS", "err_vs_ref"),
               _time_rows(cn))
    space_rows = _run_space()
    _plot(space_rows, be, cn)


if __name__ == "__main__":
    main()
