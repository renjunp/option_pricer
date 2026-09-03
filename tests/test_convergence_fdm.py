"""Convergence orders of the finite-difference schemes (shared drivers).

Temporal orders are measured against a fine-dt reference of the same scheme
on the same grid (studies.time_study), spatial order against Black-Scholes
with the spot on the strike and dt = h/4 (studies.space_study).  Expected
rates: backward Euler -> 1 in dt, Crank-Nicolson -> 2 in dt, and 2 in h.

All studies price a European call with K = 100, T = 1, r = 5%, sigma = 20%
(spot S0 = 105 off the strike for the time studies, S0 = 100 on the strike
for the space study).  The empirical order must sit within +- 0.25 of the
theoretical value on every refinement level.
"""

import pytest

from option_pricer.postprocess import convergence_rates
from option_pricer.studies import space_study, time_study


def _assert_orders(rates, order):
    """Every measured rate must lie in [order - 0.25, order + 0.25]."""
    assert len(rates) >= 3, "need at least three refinement levels"
    for rate in rates:
        assert order - 0.25 <= rate <= order + 0.25, (order, rates)


def test_backward_euler_first_order_in_time():
    rates = convergence_rates(time_study("be")["err_time"])
    _assert_orders(rates, 1.0)


def test_crank_nicolson_second_order_in_time():
    rates = convergence_rates(time_study("cn")["err_time"])
    _assert_orders(rates, 2.0)


def test_second_order_in_space():
    rates = convergence_rates(space_study()["err_bs"])
    _assert_orders(rates, 2.0)


def test_study_errors_shrink():
    """Sanity: temporal error falls with dt and spatial error falls with h.

    (The total error vs Black-Scholes of a time study need not be monotone
    in dt: at coarse dt the temporal and spatial errors can partly cancel.
    The monotone quantities are err_time vs the fine-dt reference, and the
    spatial err_bs of the space study.)
    """
    for scheme in ("be", "cn"):
        d = time_study(scheme)
        assert d["err_time"][-1] < d["err_time"][0]
        for a, b in zip(d["err_time"], d["err_time"][1:]):
            assert b < a
        assert d["err_bs"][-1] < 1e-3
    d = space_study()
    assert d["err_bs"][-1] < d["err_bs"][0]
    assert d["err_bs"][-1] < 1e-4
