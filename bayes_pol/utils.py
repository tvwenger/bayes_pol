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

from bayes_pol.ops import square_freq_rmsf
from bayes_pol.conv import convolve1d


def calc_faraday_spectrum(faraday_depth_axis, lam2_axis, stokesQ, stokesU):
    """Calculate the Faraday spectrum from Stokes Q and U via direct convolution.

    Parameters
    ----------
    faraday_depth_axis : Iterable[float]
        Faraday depth axis (rad/m2) (length F)
    lam2_axis : Iterable[float]
        Lambda^2 axis (m2) (length N)
    stokesQ : Iterable[float]
        Stokes Q data (length N)
    stokesU : Iterable[float]
        Stokes U data (length N)

    Returns
    -------
    Iterable[float]
        Complex faraday depth spectrum (length F)
    """
    arg = faraday_depth_axis[:, None] * lam2_axis
    complex_pol = stokesQ + 1.0j * stokesU
    faraday_spec = np.sum(complex_pol * np.exp(-2.0j * arg), axis=1) / len(lam2_axis)
    return faraday_spec


def calc_faraday_dispersion(
    faraday_depth_axis: Iterable[float],
    lam2_lower: Iterable[float],
    lam2_upper: Iterable[float],
    lam2_chans: Iterable[int],
    faraday_depths: Iterable[float],
    polarized_intensities: Iterable[float],
    pol_angles: Iterable[float],
    faraday_depth_fwhms: Iterable[float],
):
    """Calculate the faraday dispersion function (FDF, i.e. line profile) for each cloud
    analytically.

    Parameters
    ----------
    faraday_depth_axis : Iterable[float]
        Faraday depth axis (rad/m2) (length F)
    lam2_lower : Iterable[float]
        Lower lambda^2 limit (m2) (length W)
    lam2_upper : Iterable[float]
        Upper lambda^2 limit (m2) (length W)
    lam2_chans : Iterable[int]
        Number of lambda^2 channels in each window (length W)
    faraday_depths : Iterable[float]
        Mean Faraday depths (length C)
    polarized_intensities : Iterable[float]
        Polarized intensities (length C)
    pol_angles : Iterable[float]
        Polarization angles (length C)
    faraday_depth_fwhms : Iterable[float]
        Faraday depth FWHMs (rad/m2) (length C)

    Returns
    -------
    Iterable[float]
        Real part of the FDF (shape F x C)
    Iterable[float]
        Imaginary part of the FDF (shape F x C)
    """
    # window normalization constant (shape W)
    lam_lower = np.sqrt(lam2_lower)
    lam_upper = np.sqrt(lam2_upper)
    window_const = lam2_chans / (2.0 / lam_lower - 2.0 / lam_upper) / lam2_chans.sum()

    # Faraday depth axes offset to mean faraday depth (shape F x C x W)
    faraday_depth_axis_offset = pt.repeat(
        (faraday_depth_axis[:, None] - faraday_depths)[..., None],
        len(lam2_lower),
        axis=-1,
    )

    # Rotation measure spread function (shape F x C x W x 2)
    rmsf = square_freq_rmsf(faraday_depth_axis_offset, lam2_lower, lam2_upper)

    # Rotation measure spread function (shape F x C x W)
    rmsf_real = rmsf[..., 0]
    rmsf_imag = rmsf[..., 1]

    # cumulative polarization angle (nearest to farthest; shape C)
    cum_pol_angles = pt.cumsum(pol_angles)

    # rotate (shape F x C x W)
    rmsf_real_rotate = rmsf_real * pt.cos(
        2.0 * cum_pol_angles[None, :, None]
    ) - rmsf_imag * pt.sin(2.0 * cum_pol_angles[None, :, None])
    rmsf_imag_rotate = rmsf_imag * pt.cos(
        2.0 * cum_pol_angles[None, :, None]
    ) + rmsf_real * pt.sin(2.0 * cum_pol_angles[None, :, None])

    # Convolution kernel (shape W x C x F)
    delta_function = np.zeros_like(faraday_depth_axis)
    delta_function[len(faraday_depth_axis) // 2] = 1.0
    arg = np.sqrt(4.0 * np.pi * np.log(2.0)) * (
        pt.exp(
            -4.0
            * np.log(2.0)
            * faraday_depth_axis**2.0
            / faraday_depth_fwhms[:, None] ** 2.0
        )
        / faraday_depth_fwhms[:, None]
    )
    switch = pt.switch(
        pt.eq(faraday_depth_fwhms[:, None], 0.0),
        delta_function,
        arg,
    )
    convolution_kernel = pt.repeat(switch[None, ...], len(lam2_lower), axis=0)

    # Normalize kernel (shape W x C x F)
    convolution_kernel = (
        convolution_kernel / pt.sum(convolution_kernel, axis=-1)[..., None]
    )

    # Convolution (shape F x C x W)
    fdf_real = (
        window_const[None, None, :]
        * polarized_intensities[None, :, None]
        * convolve1d(rmsf_real_rotate.T, convolution_kernel, mode="same").T
    )
    fdf_imag = (
        window_const[None, None, :]
        * polarized_intensities[None, :, None]
        * convolve1d(rmsf_imag_rotate.T, convolution_kernel, mode="same").T
    )

    # sum over windows (shape F x C)
    fdf_real = pt.sum(fdf_real, axis=-1)
    fdf_imag = pt.sum(fdf_imag, axis=-1)
    return fdf_real, fdf_imag
