"""Fitting implementation that fits curves simply one by one in serial."""

import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from leastsqbound import leastsqbound
from scipy.optimize import least_squares


def fit_curves_mi(f, xdata, ydatas, guesses, bounds, out_pmap, use_scipy=False,
                  known_params=None):
    """Fit curves to data with multiple initializations.

    Parameters
    ----------
    f : callable
        Cost function used for fitting in form of f(parameters, x).
    xdata : ndarray, shape = [n_bvalues]
        X data points, i.e. b-values
    ydatas : ndarray, shape = [n_curves, n_bvalues]
        Y data points, i.e. signal intensity curves
    known_params : ndarray, shape = [n_curves, n_known_params]
        Optional known params for each curve
    guesses : callable
        A callable that returns an iterable of all combinations of parameter
        initializations, i.e. starting guesses, as tuples
    bounds : sequence of tuples
        Constraints for parameters, i.e. minimum and maximum values
    out_pmap : ndarray, shape = [n_curves, n_parameters+1]
        Output array

    For each signal intensity curve, the resulting parameters with best fit
    will be placed in the output array, along with an RMSE value (root mean
    square error). In case of error, curve parameters will be set to NaN and
    RMSE to infinite.

    See files fit.py and models.py for more information on usage.
    """
    if known_params is not None:
        assert len(known_params) == len(ydatas)

    for i, ydata in enumerate(tqdm.tqdm(ydatas)):
        known_param = known_params[i] if known_params is not None else []
        params, err = fit_curve_mi(f, xdata, ydata, guesses(ydata[0]), bounds,
                                   use_scipy, known_param)
        out_pmap[i, -1] = err
        if np.isfinite(err):
            out_pmap[i, :-1] = params
        else:
            out_pmap[i, :-1].fill(np.nan)


def fit_curves_mi_mp(f, xdata, ydatas, guesses, bounds, out_pmap, use_scipy=False,
                  known_params=None):
    if known_params is not None:
        assert len(known_params) == len(ydatas)

    # process_iteration = partial(
    #     fit_curve_mi,
    #     f=f, xdata=xdata, guesses=guesses, bounds=bounds,
    #     use_scipy=use_scipy, known_params=None)

    known_params = known_params if known_params is not None else []
    process_iteration = partial(process_iteration_helper,
                                f=f, xdata=xdata, guesses=guesses,
                                bounds=bounds, use_scipy=use_scipy,
                                known_params=known_params)

    results = process_map(process_iteration, ydatas, chunksize=100)

    for i, (params, err) in enumerate(results):
        out_pmap[i, -1] = err
        if np.isfinite(err):
            out_pmap[i, :-1] = params
        else:
            out_pmap[i, :-1].fill(np.nan)


def process_iteration_helper(ydata, f, xdata, guesses, bounds, use_scipy, known_params):
    guesses = guesses(ydata[0])
    return fit_curve_mi(
        f, xdata, ydata, guesses, bounds, use_scipy, known_params=known_params)


def fit_curve_mi(f, xdata, ydata, guesses, bounds, use_scipy=False,
                 known_params=[]):
    """Fit a curve to data with multiple initializations.

    Try all given combinations of parameter initializations, and return the
    parameters and RMSE of best fit.
    """
    if np.any(np.isnan(ydata)):
        return None, np.nan
    best_params = []
    best_err = np.inf

    for guess in guesses:
        if use_scipy:
            params, err = fit_curve_sp(f, xdata, ydata, guess, bounds,
                                       known_params)
        else:
            params, err = fit_curve(f, xdata, ydata, guess, bounds,
                                    known_params)
        if err < best_err:
            best_params = params
            best_err = err
    return best_params, best_err


def fit_curve(f, xdata, ydata, guess, bounds, known_params=[]):
    """Fit a curve to data."""
    def residual(p, x, y):
        return f(np.concatenate([known_params, p]), x) - y

    params, ier = leastsqbound(residual, guess, args=(xdata, ydata),
                               bounds=bounds)
    if 0 < ier < 5:
        err = rmse(residual, params, xdata, ydata)
    else:
        err = np.inf
    return params, err


def fit_curve_sp(f, xdata, ydata, guess, bounds, known_params=[]):
    """Fit a curve to data using scipy least squares."""
    def residual(p, x, y):
        return f(np.concatenate([known_params, p]), x) - y

    bounds = list(zip(*bounds))
    result = least_squares(residual, guess, args=(xdata, ydata),
                           bounds=bounds)
    if 0 < result.status < 5:
        err = rmse(residual, result.x, xdata, ydata)
    else:
        err = np.inf
    return result.x, err


def rmse(residual, p, xdata, ydata):
    """Root-mean-square error."""
    sqerr = (residual(p, xdata, ydata)) ** 2
    return np.sqrt(sqerr.mean())
