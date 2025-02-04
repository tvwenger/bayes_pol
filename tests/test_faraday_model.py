"""
test_cn_model.py
tests for CNModel

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest
import numpy as np

from bayes_spec import SpecData
from bayes_pol.faraday_model import sinc, FaradayModel


def test_sinc():
    x = np.linspace(-10.0, 10.0, 101)
    y = sinc(x).eval()
    exp = np.sin(x) / x
    exp[x == 0.0] = 1.0
    assert np.all(np.isclose(y, exp))


def test_faraday_model():
    freq_axis = np.linspace(1.1e9, 1.4e9, 300)  # Hz
    lam2_axis = (2.99792458e08 / freq_axis[::-1]) ** 2.0  # m2
    lam2_window_width = np.ptp(lam2_axis)
    faraday_axis = np.linspace(-1e5, 1e5, 525)
    data = {
        "Q": SpecData(lam2_axis, np.random.randn(len(lam2_axis)), 1.0),
        "U": SpecData(lam2_axis, np.random.randn(len(lam2_axis)), 1.0),
        "faraday_depth_abs": SpecData(faraday_axis, np.abs(np.random.randn(len(faraday_axis))), 1.0),
    }
    with pytest.raises(ValueError):
        model = FaradayModel(data, n_clouds=1)
    model = FaradayModel(data, n_clouds=1, lam2_window_width=lam2_window_width)
    model.add_priors()
    model.add_likelihood()
    assert model._validate()
