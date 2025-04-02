"""
ops.py
Custom pytensor Ops for bayes_pol

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from pytensor.gradient import DisconnectedType

from scipy.special import erfc


class SquareFreqRMSF(Op):
    """pytensor Op that evaluates the complex rotation measure spread function
    for square frequency channels.

    For input with shape N, the output has shape (N, 2) where
    the second dimension contains the real and imaginary parts.
    """

    __props__ = ()

    def output_type(self, inp):
        # add extra dim for real/imag
        return pt.TensorType(inp.dtype, shape=((None,) * inp.type.ndim) + (2,))

    def make_node(self, phi, lower, upper):
        phi = pt.as_tensor_variable(phi)
        lower = pt.as_tensor_variable(lower)
        upper = pt.as_tensor_variable(upper)
        return Apply(self, [phi, lower, upper], [self.output_type(phi)()])

    def perform(self, node, inputs, outputs):
        (phi, lower, upper) = inputs

        # catch phi = 0
        phi[phi == 0.0] += 1.0e-6
        lims = np.array([lower, upper]).T
        args = 2.0j * phi[..., None] * lims

        gammainc_half = np.sqrt(np.pi) * erfc(np.sqrt(args))
        F0 = 2.0 * (np.sqrt(args) * gammainc_half - np.exp(-args)) / np.sqrt(lims)
        out = np.zeros((*phi.shape, 2), dtype=phi.dtype)

        # F0(b) - F0(a)
        F0_diff = np.diff(F0, axis=-1)[..., 0]
        out[..., 0], out[..., 1] = np.real(F0_diff), np.imag(F0_diff)
        outputs[0][0] = out

    def grad(self, inputs, cost_grads):
        (phi, lower, upper) = inputs
        # cost grads = (d_cost/d_op[i, j],) where i = input, j = real/imag
        # this function must return (d_cost/d_x[i]) where i = input
        # where d_cost/d_x[i] = sum_j (d_cost/d_op[i, j] * d_op[i, j] / d_x[i])
        (grads,) = cost_grads
        jac = square_freq_rmsf_grad(phi, lower, upper)
        return [
            pt.sum(jac * grads, axis=-1),
            DisconnectedType()(),
            DisconnectedType()(),
        ]


class SquareFreqRMSFGrad(Op):
    """pytensor Op that evaluates the gradient of the complex rotation measure
    spread function for square frequency channels.

    For input with shape N, the output has shape (N, 2) where
    the second dimension contains the real and imaginary parts.
    """

    __props__ = ()

    def output_type(self, inp):
        # add extra dim for real/imag
        return pt.TensorType(inp.dtype, shape=((None,) * inp.type.ndim) + (2,))

    def make_node(self, phi, lower, upper):
        phi = pt.as_tensor_variable(phi)
        lower = pt.as_tensor_variable(lower)
        upper = pt.as_tensor_variable(upper)
        return Apply(self, [phi, lower, upper], [self.output_type(phi)()])

    def perform(self, node, inputs, outputs):
        (phi, lower, upper) = inputs

        # catch phi = 0
        phi[phi == 0.0] += 1.0e-6
        lims = np.array([lower, upper]).T
        args = 2.0j * phi[..., None] * lims

        gammainc_half = np.sqrt(np.pi) * erfc(np.sqrt(args))
        gradF0 = (2.0j * np.sqrt(lims) * gammainc_half) / np.sqrt(args)

        out = np.zeros((*phi.shape, 2), dtype=phi.dtype)

        # gradF0(b) - gradF0(a)
        gradF0_diff = np.diff(gradF0, axis=-1)[..., 0]
        out[..., 0], out[..., 1] = np.real(gradF0_diff), np.imag(gradF0_diff)
        outputs[0][0] = out


square_freq_rmsf = SquareFreqRMSF()
square_freq_rmsf_grad = SquareFreqRMSFGrad()
