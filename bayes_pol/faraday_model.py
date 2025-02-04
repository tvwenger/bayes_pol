"""
faraday_model.py
FaradayModel definition

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm
from pymc.distributions.transforms import CircularTransform
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel


def sinc(x: float) -> float:
    """Evaluate sin(x)/x and catch x=0.

    Parameters
    ----------
    x : float
        Position at which to evaluate

    Returns
    -------
    float
        sin(x)/x
    """
    return pt.switch(pt.eq(x, 0.0), 1.0, pt.sin(x) / x)


class FaradayModel(BaseModel):
    """Definition of the model"""

    def __init__(self, *args, lam2_window_width=None, **kwargs):
        """Initialize a new model instance"""
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Select features used for posterior clustering
        self._cluster_features += [
            "polarized_intensity",
            "faraday_depth_mean",
            "faraday_depth_fwhm",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "polarized_intensity": r"P (data brightness)",
                "faraday_depth_mean": r"$\langle F \rangle$ (rad m$^{-2}$)",
                "faraday_depth_fwhm": r"$\Delta F$ (rad m$^{-2}$)",
                "pol_angle0": r"$\phi_0$ (rad)",
            }
        )

        # save lam2 window width
        self.lam2_window_width = lam2_window_width
        if self.lam2_window_width is None:
            raise ValueError("Must supply lam2_window_width")

    def add_priors(
        self,
        prior_polarized_intensity: float = 100.0,  # data brightness
        prior_faraday_depth_mean: float = [0.0, 1000.0],  # rad m-2
        prior_faraday_depth_fwhm: float = 10.0,  # rad m-2
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_polarized_intensity : float, optional
            Prior distribution on the polarized intensity (data brightness), by default 100.0, where
            polarized_intensity ~ HalfNormal(sigma=prior)
        prior_faraday_depth_mean : Iterable[float], optional
            Prior distribution on the mean Faraday depth (rad/m2), by default [0.0, 1000.0], where
            faraday_depth_mean ~ Cauchy(alpha=prior[0], beta=prior[1])
        prior_faraday_depth_fwhm : float, optional
            Prior distribution on the Faraday depth full-width at half-maximum (rad/m2), by default 10.0, where
            faraday_depth_fwhm ~ HalfNormal(sigma=prior)
        """
        with self.model:
            # Polarized intensity (data brightness units)
            polarized_intensity_norm = pm.HalfNormal("polarized_intensity_norm", sigma=1.0, dims="cloud")
            _ = pm.Deterministic(
                "polarized_intensity", polarized_intensity_norm * prior_polarized_intensity, dims="cloud"
            )

            # Mean Faraday depth (rad m-2)
            faraday_depth_mean_norm = pm.Cauchy("faraday_depth_mean_norm", alpha=0.0, beta=1.0, dims="cloud")
            _ = pm.Deterministic(
                "faraday_depth_mean",
                prior_faraday_depth_mean[0] + prior_faraday_depth_mean[1] * faraday_depth_mean_norm,
                dims="cloud",
            )

            # FWHM Faraday depth (rad m-2)
            faraday_depth_fwhm_norm = pm.HalfNormal("faraday_depth_fwhm_norm", sigma=1.0, dims="cloud")
            _ = pm.Deterministic("faraday_depth_fwhm", faraday_depth_fwhm_norm * prior_faraday_depth_fwhm, dims="cloud")

            # Polarization angle at lambda = 0 (rad; shape: clouds)
            pol_angle0_norm = pm.Uniform(
                "pol_angle0_norm",
                lower=-np.pi,
                upper=np.pi,
                dims="cloud",
                transform=CircularTransform(),
            )
            _ = pm.Deterministic("pol_angle0", 0.5 * pol_angle0_norm, dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "Q", "U", and "faraday_depth_abs".
        Spectral units for "Q" and "U" should be square wavelength in m2.
        Spectral units for "faraday_depth_abs" should be rad/m2.
        Order of clouds is nearest to farthest.
        """
        with self.model:
            # Predict Faraday depth spectrum (shape: spectral)
            faraday_depth_abs = pt.abs(
                self.model["polarized_intensity"]
                * sinc(
                    (self.data["faraday_depth_abs"].spectral[:, None] - self.model["faraday_depth_mean"])
                    * self.lam2_window_width
                )
            ).sum(axis=1)

            _ = pm.TruncatedNormal(
                "faraday_depth_abs",
                mu=faraday_depth_abs,
                lower=0.0,
                sigma=self.data["faraday_depth_abs"].noise,
                observed=self.data["faraday_depth_abs"].brightness,
            )

            # Predict Stokes Q and U, sum over clouds (shape: spectral)
            stokesQ = (
                self.model["polarized_intensity"]
                * pt.exp(
                    -self.model["faraday_depth_fwhm"] ** 2.0
                    * self.data["Q"].spectral[:, None] ** 2.0
                    / (4.0 * np.log(2.0))
                )
                * pt.cos(
                    2.0
                    * (
                        pt.cumsum(self.model["pol_angle0"])
                        + self.model["faraday_depth_mean"] * self.data["Q"].spectral[:, None]
                    )
                )
            ).sum(axis=1)
            stokesU = (
                self.model["polarized_intensity"]
                * pt.exp(
                    -self.model["faraday_depth_fwhm"] ** 2.0
                    * self.data["U"].spectral[:, None] ** 2.0
                    / (4.0 * np.log(2.0))
                )
                * pt.sin(
                    2.0
                    * (
                        pt.cumsum(self.model["pol_angle0"])
                        + self.model["faraday_depth_mean"] * self.data["U"].spectral[:, None]
                    )
                )
            ).sum(axis=1)

            _ = pm.Normal("Q", mu=stokesQ, sigma=self.data["Q"].noise, observed=self.data["Q"].brightness)
            _ = pm.Normal("U", mu=stokesU, sigma=self.data["U"].noise, observed=self.data["U"].brightness)
