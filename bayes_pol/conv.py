"""
conv.py
1-D convolution pytensor implementation taken from
https://github.com/pymc-devs/pytensor/blob/main/pytensor/tensor/signal/conv.py

Copyright (c) 2008-2019, Theano Development Team
Copyright (c) 2020-2021, PyMC Development team
Copyright (c) 2021-2022, Aesara Development Team
Copyright (c) 2022, PyMC Development team
All rights reserved.

Contains code from NumPy, Copyright (c) 2005-2016, NumPy Developers.
Contains code from Aesara, Copyright (c) 2021-2022, Aesara Developers.
All rights reserved.

theano/scan/*.py[c]: Razvan Pascanu, Frederic Bastien, James Bergstra, Pascal Lamblin, Arnaud Bergeron, PyMC Developers, PyTensor Developers, (c) 2010, Universite de Montreal
theano/tensor/sharedvar.py: James Bergstra, (c) 2010, Universite de Montreal, 3-clause BSD License
theano/gradient.py: James Bergstra, Razvan Pascanu, Arnaud Bergeron, Ian Goodfellow, PyMC Developers, PyTensor Developers, (c) 2011, Universite de Montreal, 3-clause BSD License
theano/compile/monitormode.py: this code was initially copied from the 'pyutools' package by its original author, and re-licensed under Theano's license.

Contains frozendict code from slezica’s python-frozendict(https://github.com/slezica/python-frozendict/blob/master/frozendict/__init__.py), Copyright (c) 2012 Santiago Lezica. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of PyTensor, Theano, nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import TYPE_CHECKING, Literal, cast

from numpy import convolve as numpy_convolve

from pytensor.graph import Apply, Op
from pytensor.scalar.basic import upcast
from pytensor.tensor.basic import as_tensor_variable, join, zeros
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import maximum, minimum
from pytensor.tensor.type import vector
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class Conv1d(Op):
    __props__ = ("mode",)
    gufunc_signature = "(n),(k)->(o)"

    def __init__(self, mode: Literal["full", "valid"] = "full"):
        if mode not in ("full", "valid"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def make_node(self, in1, in2):
        in1 = as_tensor_variable(in1)
        in2 = as_tensor_variable(in2)

        assert in1.ndim == 1
        assert in2.ndim == 1

        dtype = upcast(in1.dtype, in2.dtype)

        n = in1.type.shape[0]
        k = in2.type.shape[0]

        if n is None or k is None:
            out_shape = (None,)
        elif self.mode == "full":
            out_shape = (n + k - 1,)
        else:  # mode == "valid":
            out_shape = (max(n, k) - min(n, k) + 1,)

        out = vector(dtype=dtype, shape=out_shape)
        return Apply(self, [in1, in2], [out])

    def perform(self, node, inputs, outputs):
        # We use numpy_convolve as that's what scipy would use if method="direct" was passed.
        # And mode != "same", which this Op doesn't cover anyway.
        outputs[0][0] = numpy_convolve(*inputs, mode=self.mode)

    def infer_shape(self, fgraph, node, shapes):
        in1_shape, in2_shape = shapes
        n = in1_shape[0]
        k = in2_shape[0]
        if self.mode == "full":
            shape = n + k - 1
        else:  # mode == "valid":
            shape = maximum(n, k) - minimum(n, k) + 1
        return [[shape]]

    def L_op(self, inputs, outputs, output_grads):
        in1, in2 = inputs
        [grad] = output_grads

        if self.mode == "full":
            valid_conv = type(self)(mode="valid")
            in1_bar = valid_conv(grad, in2[::-1])
            in2_bar = valid_conv(grad, in1[::-1])

        else:  # mode == "valid":
            full_conv = type(self)(mode="full")
            n = in1.shape[0]
            k = in2.shape[0]
            kmn = maximum(0, k - n)
            nkm = maximum(0, n - k)
            # We need mode="full" if k >= n else "valid" for `in1_bar` (opposite for `in2_bar`), but mode is not symbolic.
            # Instead, we always use mode="full" and slice the result so it behaves like "valid" for the input that's shorter.
            in1_bar = full_conv(grad, in2[::-1])
            in1_bar = in1_bar[kmn : in1_bar.shape[0] - kmn]
            in2_bar = full_conv(grad, in1[::-1])
            in2_bar = in2_bar[nkm : in2_bar.shape[0] - nkm]

        return [in1_bar, in2_bar]


def convolve1d(
    in1: "TensorLike",
    in2: "TensorLike",
    mode: Literal["full", "valid", "same"] = "full",
) -> TensorVariable:
    """Convolve two one-dimensional arrays.

    Convolve in1 and in2, with the output size determined by the mode argument.

    Parameters
    ----------
    in1 : (..., N,) tensor_like
        First input.
    in2 : (..., M,) tensor_like
        Second input.
    mode : {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        - 'full': The output is the full discrete linear convolution of the inputs, with shape (..., N+M-1,).
        - 'valid': The output consists only of elements that do not rely on zero-padding, with shape (..., max(N, M) - min(N, M) + 1,).
        - 'same': The output is the same size as in1, centered with respect to the 'full' output.

    Returns
    -------
    out: tensor_variable
        The discrete linear convolution of in1 with in2.

    """
    in1 = as_tensor_variable(in1)
    in2 = as_tensor_variable(in2)

    if mode == "same":
        # We implement "same" as "valid" with padded `in1`.
        in1_batch_shape = tuple(in1.shape)[:-1]
        zeros_left = in2.shape[-1] // 2
        zeros_right = (in2.shape[-1] - 1) // 2
        in1 = join(
            -1,
            zeros((*in1_batch_shape, zeros_left), dtype=in2.dtype),
            in1,
            zeros((*in1_batch_shape, zeros_right), dtype=in2.dtype),
        )
        mode = "valid"

    return cast(TensorVariable, Blockwise(Conv1d(mode=mode))(in1, in2))
