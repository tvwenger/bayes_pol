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
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

from scipy.special import erfc


class ErfcOp(Op):
    """pytensor Op that evaluates: erfc(sqrt(-1.0j*x))

    For input with shape N, the output has shape (N, 2) where
    the second dimension contains the real and imaginary parts.
    """

    __props__ = ()

    def output_type(self, inp):
        # add extra dim for real/imag
        return pt.TensorType(inp.dtype, shape=((None,) * inp.type.ndim) + (2,))

    def make_node(self, inp):
        inp = pt.as_tensor_variable(inp)
        return Apply(self, [inp], [self.output_type(inp)()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        res = erfc(np.sqrt(-1.0j * x))
        out = np.zeros((*x.shape, 2), dtype=x.dtype)
        out[..., 0], out[..., 1] = np.real(res), np.imag(res)
        outputs[0][0] = out

    def grad(self, inputs, cost_grads):
        (x,) = inputs
        # cost grads = (d_cost/d_op[i, j],) where i = input, j = real/imag
        # this function must return (d_cost/d_x[i]) where i = input
        # where d_cost/d_x[i] = sum_j (d_cost/d_op[i, j] * d_op[i, j] / d_x[i])
        (grads,) = cost_grads

        jac = pt.stack(
            [
                # real part
                pt.switch(
                    pt.eq(x, 0.0),
                    0.0,
                    pt.switch(
                        pt.gt(x, 0.0),
                        -(pt.cos(x) + pt.sin(x)) / pt.sqrt(2.0 * np.pi * x),
                        (pt.cos(x) - pt.sin(x)) / pt.sqrt(-2.0 * np.pi * x),
                    ),
                ),
                # imag part
                pt.switch(
                    pt.eq(x, 0.0),
                    1.0,
                    pt.switch(
                        pt.gt(x, 0.0),
                        (pt.cos(x) - pt.sin(x)) / pt.sqrt(2.0 * np.pi * x),
                        (pt.cos(x) + pt.sin(x)) / pt.sqrt(-2.0 * np.pi * x),
                    ),
                ),
            ],
            axis=-1,
        )
        return [pt.sum(jac * cost_grads[0], axis=-1)]


erfc_func = ErfcOp()


def calc_faraday_depth(faraday_depth_axis, lam2_axis, stokesQ, stokesU):
    """Calculate the Faraday depth spectrum from Stokes Q and U

    Parameters
    ----------
    faraday_depth_axis : Iterable[float]
        Faraday depth axis (rad/m2) (length F)
    lam2_axis : Iterable[float]
        Lambda^2 axis (m2) (length N)
    stokesQ : Iterable[float]
        Stokes Q data (length N)
    stokesU : Iterable[float]
        Stokes U data (length N)P

    Returns
    -------
    Iterable[float]
        Complex faraday depth spectrum (length F)
    """
    arg = faraday_depth_axis[:, None] * lam2_axis
    complex_pol = stokesQ + 1.0j * stokesU
    faraday_spec = np.sum(complex_pol * np.exp(-2.0j * arg), axis=1) / len(lam2_axis)
    return faraday_spec


def calc_rmsf(
    faraday_depth_axis: Iterable[float],
    lam2_lower: Iterable[float],
    lam2_upper: Iterable[float],
    lam2_chans: Iterable[int],
    faraday_depths: Iterable[float],
    polarized_intensities: Iterable[float],
    pol_angles: Iterable[float],
):
    """Calculate the rotation measure spread function per cloud.

    Parameters
    ----------
    faraday_depth_axis : Iterable[float]
        Faraday depth axis (rad/m2) (length F)
    lam2_lower : Iterable[float]
        Lower lambda^2 limit (m2) (length W)
    lam2_upper : Iterable[float]
        Upper lambda^2 limit (m2) (length W)
    lam2_chans : Iterable[int]
        Number of channels per window (length W)
    faraday_depths : Iterable[float]
        Mean Faraday depths (length C)
    polarized_intensities : Iterable[float]
        Polarized intensities (length C)
    pol_angles : Iterable[float]
        Polarization angles (length C)

    Returns
    -------
    Iterable[float]
        Real part of the RMSF (shape F x C)
    Iterable[float]
        Imaginary part of the RMSF (shape F x C)
    """
    # window constant (shape W)
    lam_lower = np.sqrt(lam2_lower)
    lam_upper = np.sqrt(lam2_upper)
    const = lam2_chans / (1.0 / lam_lower - 1.0 / lam_upper) / lam2_chans.sum()

    # Faraday depth axes offset to mean faraday depth (shape F x C)
    faraday_depth_axis_offset = faraday_depth_axis[:, None] - faraday_depths

    # cumulative polarization angle (clouds ordered from nearest to farthest)
    cum_pol_angles = pt.cumsum(pol_angles)

    # erfc terms (shape F x C x W x 2)
    erfc_upper = erfc_func(2.0 * (faraday_depth_axis_offset[..., None] * lam2_upper))
    erfc_lower = erfc_func(2.0 * (faraday_depth_axis_offset[..., None] * lam2_lower))

    # shape (F x C x W)
    re_erfc_upper = pt.switch(
        pt.gt(faraday_depth_axis_offset[..., None], 0.0),
        pt.sqrt(np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_upper[..., 0] + erfc_upper[..., 1]),
        pt.sqrt(-np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_upper[..., 0] - erfc_upper[..., 1]),
    )
    im_erfc_upper = pt.switch(
        pt.gt(faraday_depth_axis_offset[..., None], 0.0),
        pt.sqrt(np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_upper[..., 1] - erfc_upper[..., 0]),
        pt.sqrt(-np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_upper[..., 1] + erfc_upper[..., 0]),
    )
    re_erfc_lower = pt.switch(
        pt.gt(faraday_depth_axis_offset[..., None], 0.0),
        pt.sqrt(np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_lower[..., 0] + erfc_lower[..., 1]),
        pt.sqrt(-np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_lower[..., 0] - erfc_lower[..., 1]),
    )
    im_erfc_lower = pt.switch(
        pt.gt(faraday_depth_axis_offset[..., None], 0.0),
        pt.sqrt(np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_lower[..., 1] - erfc_lower[..., 0]),
        pt.sqrt(-np.pi * faraday_depth_axis_offset[..., None])
        * (erfc_lower[..., 1] + erfc_lower[..., 0]),
    )

    # shape (F x C x W)
    re_erfc_diff = re_erfc_upper - re_erfc_lower
    im_erfc_diff = im_erfc_upper - im_erfc_lower

    # exponential terms (shape F x C x W)
    re_exp_diff = (
        pt.cos(2.0 * (lam2_lower * faraday_depth_axis_offset[..., None])) / lam_lower
        - pt.cos(2.0 * (lam2_upper * faraday_depth_axis_offset[..., None])) / lam_upper
    )
    im_exp_diff = (
        pt.sin(2.0 * (lam2_lower * faraday_depth_axis_offset[..., None])) / lam_lower
        - pt.sin(2.0 * (lam2_upper * faraday_depth_axis_offset[..., None])) / lam_upper
    )

    # rmsf per cloud (shape F x C x W)
    re_rmsf_cloud = (
        const[None, None, :]
        * polarized_intensities[None, :, None]
        * (re_exp_diff + re_erfc_diff)
    )
    im_rmsf_cloud = (
        const[None, None, :]
        * polarized_intensities[None, :, None]
        * (im_exp_diff + im_erfc_diff)
    )

    # rotate (shape F x C x W)
    re_rmsf_cloud_rotate = re_rmsf_cloud * pt.cos(
        -2.0 * cum_pol_angles[None, :, None]
    ) - im_rmsf_cloud * pt.sin(-2.0 * cum_pol_angles[None, :, None])
    im_rmsf_cloud_rotate = im_rmsf_cloud * pt.cos(
        -2.0 * cum_pol_angles[None, :, None]
    ) + re_rmsf_cloud * pt.sin(-2.0 * cum_pol_angles[None, :, None])

    # sum over windows (shape F x C)
    re_rmsf = pt.sum(re_rmsf_cloud_rotate, axis=-1)
    im_rmsf = pt.sum(im_rmsf_cloud_rotate, axis=-1)
    return re_rmsf, im_rmsf
