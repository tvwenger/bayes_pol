"""
utils.py
bayes_pol utility functions

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import numpy as np
import pytensor.tensor as pt


def get_faraday_depth_params(
    lam2_axis: Iterable[float],
    faraday_depth_axis: Iterable[float],
):
    """Calculate constants used by calc_faraday_abs

    Parameters
    ----------
    lam2_axis : Iterable[float]
        Wavelength^2 axis (m2)
    faraday_depth_axis : Iterable[float]
        Faraday depth axis (rad/m2)

    Returns
    -------
    Iterable[float]
        cos_spectral, sin_spectral, a, b, c, len(lam2_axis)
        Parameters used by calc_faraday_depth_abs
    """
    cos_spectral = np.cos(faraday_depth_axis[:, None] * lam2_axis)
    sin_spectral = np.sin(faraday_depth_axis[:, None] * lam2_axis)
    a = np.sum(cos_spectral**2.0, axis=1) + np.sum(sin_spectral**2.0, axis=1)
    b = np.sum(sin_spectral**2.0, axis=1) + np.sum(cos_spectral**2.0, axis=1)
    c = np.sum(cos_spectral * sin_spectral, axis=1) - np.sum(sin_spectral * cos_spectral, axis=1)
    return cos_spectral, sin_spectral, a, b, c, len(lam2_axis)


def calc_faraday_depth_abs(
    stokesQ: Iterable[float],
    stokesU: Iterable[float],
    *params,
):
    """Calculate the Faraday depth spectrum for non-uniform lambda^2 samples
    See: https://bayes.wustl.edu/glb/trans.pdf

    Parameters
    ----------
    stokesQ : Iterable[float]
        Stokes Q data (real part of complex polarization)
    stokesU : Iterable[float]
        Stokes U data (imaginary part of complex polarization)
    params : float
        cos_spectral, sin_spectral, a, b, c
        Parameters returned by get_faraday_depth_params

    Returns
    -------
    Iterable[float]
        Magnitude of faraday depth spectrum (same units as Stokes Q and U)
    """
    cos_spectral, sin_spectral, a, b, c, len_lam2_axis = params
    T1 = pt.sum(stokesQ * cos_spectral, axis=1) + pt.sum(stokesU * sin_spectral, axis=1)
    T2 = pt.sum(stokesQ * sin_spectral, axis=1) - pt.sum(stokesU * cos_spectral, axis=1)
    return pt.sqrt((b * T1**2.0 + a * T2**2.0 - 2.0 * c * T1 * T2) / (a * b - c**2.0)) / np.sqrt(len_lam2_axis)
