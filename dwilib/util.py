"""Utility functionality."""

import numpy as np


def normalize_si_curve(si):
    """Normalize a signal intensity curve (divide all by the first value).

    Note that this function does not manage error cases where the first value
    is zero or the curve rises at some point. See normalize_si_curve_fix().
    """
    assert si.ndim == 1
    si[:] /= si[0]


def normalize_si_curve_fix(si):
    """Normalize a signal intensity curve (divide all by the first value).

    This version handles some error cases. If the first value is zero, all
    values are just set to zero. If any value is higher than the previous one,
    it is set to the same value (curves are never supposed to rise).
    """
    assert si.ndim == 1
    if si[0] == 0:
        si[:] = 0
    else:
        for i in range(1, len(si)):
            if si[i] > si[i - 1]:
                si[i] = si[i - 1]
        si[:] /= si[0]

