"""Error measures and convergence rates against a reference (here the
Black-Scholes closed form)."""

import math


def relative_error(approx, exact):
    """Relative error ``|approx - exact| / |exact|`` of a positive price."""
    return abs(approx - exact) / abs(exact)


def convergence_rates(errors):
    """Empirical orders between successive grid refinements.

    ``errors`` at halved step sizes dt (or h); each entry is
    ``log2(errors[i] / errors[i + 1])``, i.e. the order measured between
    two consecutive levels.
    """
    if len(errors) < 2:
        return []
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
