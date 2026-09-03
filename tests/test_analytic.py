"""Black-Scholes closed form: reference values, parity and asymptotics.

Smoke reference (S = K = 100, T = 1, r = 5%, sigma = 20%): the at-the-money
European call is 10.450583572185565 (widely quoted textbook value); the put
follows from put-call parity, C - P = S - K e^{-rT}.
"""

import math

import pytest

from option_pricer.analytic import (
    bs_delta,
    bs_gamma,
    bs_greeks,
    bs_price,
    bs_vega,
    parity_mismatch,
)

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
CALL_REF = 10.450583572185565
PUT_REF = 5.573526022256971


def test_smoke_call_reference():
    """The ATM call must reproduce the reference 10.4506... value."""
    assert abs(bs_price(S0, K, T, R, SIGMA, "call") - CALL_REF) < 1e-9


def test_smoke_put_reference():
    assert abs(bs_price(S0, K, T, R, SIGMA, "put") - PUT_REF) < 1e-9


def test_put_call_parity():
    c = bs_price(S0, K, T, R, SIGMA, "call")
    p = bs_price(S0, K, T, R, SIGMA, "put")
    assert abs((c - p) - (S0 - K * math.exp(-R * T))) < 1e-9
    assert parity_mismatch(c, p, S0, K, T, R) < 1e-9


def test_expiry_limit_recovers_payoff():
    """As T -> 0 the price collapses onto the payoff max(S-K, 0)."""
    eps = 1e-9
    assert abs(bs_price(120.0, 100.0, eps, R, SIGMA, "call") - 20.0) < 1e-3
    assert bs_price(80.0, 100.0, eps, R, SIGMA, "call") < 1e-6
    assert bs_price(80.0, 100.0, eps, R, SIGMA, "put") - 20.0 < 1e-3


def test_zero_spot_limit():
    """As S -> 0 the call is worthless and the put is the discounted strike."""
    s = 1e-12
    assert abs(bs_price(s, K, T, R, SIGMA, "call")) < 1e-9
    assert abs(bs_price(s, K, T, R, SIGMA, "put")
               - K * math.exp(-R * T)) < 1e-6


def test_greeks_signs_and_parity():
    d_c = bs_delta(S0, K, T, R, SIGMA, "call")
    d_p = bs_delta(S0, K, T, R, SIGMA, "put")
    g_c = bs_gamma(S0, K, T, R, SIGMA, "call")
    g_p = bs_gamma(S0, K, T, R, SIGMA, "put")
    v_c = bs_vega(S0, K, T, R, SIGMA, "call")
    assert 0.0 < d_c < 1.0
    assert -1.0 < d_p < 0.0
    assert abs((d_c - d_p) - 1.0) < 1e-12
    assert abs(g_c - g_p) < 1e-15
    assert abs(v_c - bs_vega(S0, K, T, R, SIGMA, "put")) < 1e-12
    assert g_c > 0.0 and v_c > 0.0


def test_greeks_agree_with_numerical_differences():
    """Closed-form Greeks must match central differences of bs_price."""
    e_s = 0.1  # delta needs a small bump: |C'''| e^2/6 ~ 1e-6
    c_up = bs_price(S0 + e_s, K, T, R, SIGMA, "call")
    c_dn = bs_price(S0 - e_s, K, T, R, SIGMA, "call")
    delta_num = (c_up - c_dn) / (2.0 * e_s)
    assert abs(delta_num - bs_delta(S0, K, T, R, SIGMA, "call")) < 1e-5

    e_g = 5.0  # gamma needs a wide-enough bump to beat round-off
    c_g_up = bs_price(S0 + e_g, K, T, R, SIGMA, "call")
    c_g_dn = bs_price(S0 - e_g, K, T, R, SIGMA, "call")
    gamma_num = (c_g_up - 2.0 * CALL_REF + c_g_dn) / (e_g * e_g)
    assert abs(gamma_num - bs_gamma(S0, K, T, R, SIGMA, "call")) < 2e-4

    e_v = 2e-3
    vega_num = (bs_price(S0, K, T, R, SIGMA + e_v, "call")
                - bs_price(S0, K, T, R, SIGMA - e_v, "call")) / (2.0 * e_v)
    assert abs(vega_num - bs_vega(S0, K, T, R, SIGMA, "call")) < 2e-4


def test_greeks_tuple_matches_individual():
    assert bs_greeks(S0, K, T, R, SIGMA, "call") == pytest.approx(
        (bs_delta(S0, K, T, R, SIGMA, "call"),
         bs_gamma(S0, K, T, R, SIGMA, "call"),
         bs_vega(S0, K, T, R, SIGMA, "call")))
