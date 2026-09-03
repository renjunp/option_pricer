"""Terminal payoff functions: kink, asymptotes and input validation.

    call payoff: max(S - K, 0)  (zero at/under the strike, linear above)
    put payoff:  max(K - S, 0)  (linear below the strike, zero at/above)
"""

import numpy as np
import pytest

from option_pricer.payoff import payoff


def test_call_payoff_kink():
    assert payoff(100.0, 100.0, "call") == 0.0
    assert payoff(99.0, 100.0, "call") == 0.0
    assert payoff(120.0, 100.0, "call") == 20.0


def test_put_payoff_kink():
    assert payoff(100.0, 100.0, "put") == 0.0
    assert payoff(101.0, 100.0, "put") == 0.0
    assert payoff(80.0, 100.0, "put") == 20.0


def test_payoff_vectorized():
    S = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(payoff(S, 100.0, "call"), [0.0, 0.0, 10.0])
    np.testing.assert_allclose(payoff(S, 100.0, "put"), [10.0, 0.0, 0.0])


def test_payoff_zero_spot_and_deep_limits():
    assert payoff(0.0, 100.0, "call") == 0.0
    assert payoff(0.0, 100.0, "put") == 100.0
    assert payoff(1e6, 100.0, "call") == pytest.approx(1e6 - 100.0)


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        payoff(100.0, 100.0, "digital")
