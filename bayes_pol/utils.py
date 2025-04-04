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


def construct_faraday_spectrum(faraday_depth_axis, lam2_axis, stokesQ, stokesU):
    """Construct the Faraday spectrum from Stokes Q and U via direct convolution.

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
    arg = -2.0 * faraday_depth_axis[:, None] * lam2_axis
    faraday_spec_real = pt.sum(
        stokesQ * pt.cos(arg) - stokesU * pt.sin(arg), axis=1
    ) / len(lam2_axis)
    faraday_spec_imag = pt.sum(
        stokesU * pt.cos(arg) + stokesQ * pt.sin(arg), axis=1
    ) / len(lam2_axis)
    return faraday_spec_real, faraday_spec_imag


def construct_rmsf(faraday_depth_axis, lam2_lower, lam2_upper):
    """Construct the rotation measure spread function (RMSF) assuming square
    channels in frequency space.

    Parameters
    ----------
    faraday_depth_axis : Iterable[float]
        Faraday depth axis (rad/m2) (length F)
    lam2_lower : Iterable[float]
        Lower lambda^2 limit (m2) (length W)
    lam2_upper : Iterable[float]
        Upper lambda^2 limit (m2) (length W)

    Returns
    -------
    Iterable[float]
        Real part of the RMSF (shape F)
    Iterable[float]
        Imaginary part of the RMSF (shape F)
    """
    # Rotation measure spread function (shape F x W x 2)
    rmsf = square_freq_rmsf(faraday_depth_axis[:, None], lam2_lower, lam2_upper).eval()

    # Rotation measure spread function (shape F x W)
    rmsf_real = rmsf[..., 0]
    rmsf_imag = rmsf[..., 1]

    # Sum over window
    return rmsf_real.sum(axis=-1), rmsf_imag.sum(axis=-1)


def predict_stokes_QU(
    lam2_axis: Iterable[float],
    polarization_fraction: Iterable[float],
    faraday_depth_mean: Iterable[float],
    faraday_depth_fwhm: Iterable[float],
    pol_angle0: Iterable[float],
):
    """Predict Stokes Q and U from model parameters.

    Parameters
    ----------
    lam2_axis : Iterable[float]
        Wavelength^2 axis (m2) (length N)
    polarization_fraction : Iterable[float]
        Polarization fraction (length C)
    faraday_depth_mean : Iterable[float]
        Mean Faraday depth (rad/m2; length C)
    faraday_depth_fwhm : Iterable[float]
        FWHM Faraday depth (rad/m2; length C)
    pol_angle0 : Iterable[float]
        Polarization angle at lambda=0 (rad; length C)

    Returns
    -------
    Iterable[float]
        Stokes Q (shape N x C)
    Iterable[float]
        Stokes U (shape N x C)
    """
    depol = pt.exp(
        -(faraday_depth_fwhm**2.0) * lam2_axis[:, None] ** 2.0 / (4.0 * np.log(2.0))
    )
    arg = 2.0 * (pol_angle0 + faraday_depth_mean * lam2_axis[:, None])
    stokesQ = polarization_fraction * depol * pt.cos(arg)
    stokesU = polarization_fraction * depol * pt.sin(arg)
    return stokesQ, stokesU


def predict_faraday_dispersion(
    faraday_depth_axis: Iterable[float],
    polarization_fraction: Iterable[float],
    faraday_depth_mean: Iterable[float],
    faraday_depth_fwhm: Iterable[float],
    pol_angle0: Iterable[float],
    lam2_lower: Iterable[float],
    lam2_upper: Iterable[float],
    lam2_chans: Iterable[int],
):
    """Calculate the Faraday dispersion function (FDF, i.e. line profile) for each cloud
    analytically.

    Parameters
    ----------
    faraday_depth_axis : Iterable[float]
        Faraday depth axis (rad/m2) (length F)
    polarization_fraction : Iterable[float]
        Polarization fraction (length C)
    faraday_depth_mean : Iterable[float]
        Mean Faraday depth (rad/m2; length C)
    faraday_depth_fwhm : Iterable[float]
        FWHM Faraday depth (rad/m2; length C)
    pol_angle0 : Iterable[float]
        Polarization angle at lambda=0 (rad; length C)
    lam2_lower : Iterable[float]
        Lower lambda^2 limit (m2) (length W)
    lam2_upper : Iterable[float]
        Upper lambda^2 limit (m2) (length W)
    lam2_chans : Iterable[int]
        Number of lambda^2 channels in each window (length W)

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
        (faraday_depth_axis[:, None] - faraday_depth_mean)[..., None],
        len(lam2_lower),
        axis=-1,
    )

    # Rotation measure spread function (shape F x C x W x 2)
    rmsf = window_const[None, None, :, None] * square_freq_rmsf(
        faraday_depth_axis_offset, lam2_lower, lam2_upper
    )

    # Rotation measure spread function (shape F x C)
    rmsf_real = rmsf[..., 0].sum(axis=-1)
    rmsf_imag = rmsf[..., 1].sum(axis=-1)

    # rotate (shape F x C)
    arg = -2.0 * pol_angle0[None, :]
    rmsf_real_rotate = rmsf_real * pt.cos(arg) + rmsf_imag * pt.sin(arg)
    rmsf_imag_rotate = rmsf_imag * pt.cos(arg) - rmsf_real * pt.sin(arg)

    # Convolution kernel (shape C x F)
    delta_function = np.zeros_like(faraday_depth_axis)
    delta_function[len(faraday_depth_axis) // 2] = 1.0
    faraday_depth_fwhm2 = faraday_depth_fwhm**2.0
    arg = np.sqrt(4.0 * np.pi * np.log(2.0) / faraday_depth_fwhm2[:, None]) * pt.exp(
        -4.0 * np.log(2.0) * faraday_depth_axis**2.0 / faraday_depth_fwhm2[:, None]
    )
    convolution_kernel = pt.switch(
        pt.eq(faraday_depth_fwhm2[:, None], 0.0),
        delta_function,
        arg,
    )

    # Normalize kernel (shape C x F)
    convolution_kernel = (
        convolution_kernel / pt.sum(convolution_kernel, axis=-1)[..., None]
    )

    # Convolution (shape F x C)
    fdf_real = (
        polarization_fraction[None, :]
        * convolve1d(rmsf_real_rotate.T, convolution_kernel, mode="same").T
    )
    fdf_imag = (
        polarization_fraction[None, :]
        * convolve1d(rmsf_imag_rotate.T, convolution_kernel, mode="same").T
    )
    return fdf_real, fdf_imag
