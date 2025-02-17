"""
faraday_model.py
FaradayModel definition

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import pymc as pm
from pymc.distributions.transforms import CircularTransform
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel

from bayes_pol.utils import calc_rmsf


class FaradayModel(BaseModel):
    """Definition of the model"""

    def __init__(self, *args, **kwargs):
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
                "polarized_intensity": r"P (\%)",
                "faraday_depth_mean": r"$\langle F \rangle$ (rad m$^{-2}$)",
                "faraday_depth_fwhm": r"$\Delta F$ (rad m$^{-2}$)",
                "pol_angle0": r"$\phi_0$ (rad)",
            }
        )

        # Get upper and lower frequency and lambda^2 limits of windows
        self.lam2_lower = np.array(
            [dataset.spectral.min() for key, dataset in self.data.items() if "Q" in key]
        )
        self.lam2_upper = np.array(
            [dataset.spectral.max() for key, dataset in self.data.items() if "Q" in key]
        )
        self.num_chans = np.array(
            [len(dataset.spectral) for key, dataset in self.data.items() if "Q" in key]
        )

    def add_priors(
        self,
        prior_faraday_depth_mean: Iterable[float] = [0.0, 1000.0],  # rad m-2
        prior_faraday_depth_fwhm: float = 10.0,  # rad m-2
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_faraday_depth_mean : Iterable[float], optional
            Prior distribution on the mean Faraday depth (rad/m2), by default [0.0, 1000.0], where
            faraday_depth_mean ~ Cauchy(alpha=prior[0], beta=prior[1])
        prior_faraday_depth_fwhm : float, optional
            Prior distribution on the Faraday depth full-width at half-maximum (rad/m2), by default 10.0, where
            faraday_depth_fwhm ~ HalfNormal(sigma=prior)
        """
        with self.model:
            # Polarized intensity (data brightness units)
            _ = pm.Beta("polarized_intensity", alpha=2.0, beta=2.0, dims="cloud")

            # Mean Faraday depth (rad m-2)
            faraday_depth_mean_norm = pm.Cauchy(
                "faraday_depth_mean_norm", alpha=0.0, beta=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "faraday_depth_mean",
                prior_faraday_depth_mean[0]
                + prior_faraday_depth_mean[1] * faraday_depth_mean_norm,
                dims="cloud",
            )

            # FWHM Faraday depth (rad m-2)
            faraday_depth_fwhm_norm = pm.HalfNormal(
                "faraday_depth_fwhm_norm", sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "faraday_depth_fwhm",
                faraday_depth_fwhm_norm * prior_faraday_depth_fwhm,
                dims="cloud",
            )

            # Polarization angle at lambda = 0 (rad; shape: clouds)
            pol_angle0_norm = pm.Uniform(
                "pol_angle0_norm",
                lower=-np.pi,
                upper=np.pi,
                dims="cloud",
                transform=CircularTransform(),
            )
            _ = pm.Deterministic("pol_angle0", 0.5 * pol_angle0_norm, dims="cloud")

            # Faraday depth rms (data brightness)
            faraday_depth_sigma_norm = pm.HalfNormal(
                "faraday_depth_sigma_norm", sigma=1.0
            )
            _ = pm.Deterministic(
                "faraday_depth_sigma",
                faraday_depth_sigma_norm / np.sqrt(np.sum(self.num_chans)),
            )

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "Q", "U", and "faraday_depth_abs".
        Spectral units for "Q" and "U" should be square wavelength in m2.
        Spectral units for "faraday_depth_abs" should be rad/m2.
        Order of clouds is nearest to farthest.
        """
        with self.model:
            # Predict Stokes Q and U, sum over clouds (shape: spectral, clouds)
            for key in self.data.keys():
                if "Q" in key:
                    stokes = (
                        self.model["polarized_intensity"]
                        * pt.exp(
                            -self.model["faraday_depth_fwhm"] ** 2.0
                            * self.data[key].spectral[:, None] ** 2.0
                            / (4.0 * np.log(2.0))
                        )
                        * pt.cos(
                            2.0
                            * (
                                pt.cumsum(self.model["pol_angle0"])
                                + self.model["faraday_depth_mean"]
                                * self.data[key].spectral[:, None]
                            )
                        )
                    )
                elif "U" in key:
                    stokes = (
                        self.model["polarized_intensity"]
                        * pt.exp(
                            -self.model["faraday_depth_fwhm"] ** 2.0
                            * self.data[key].spectral[:, None] ** 2.0
                            / (4.0 * np.log(2.0))
                        )
                        * pt.sin(
                            2.0
                            * (
                                pt.cumsum(self.model["pol_angle0"])
                                + self.model["faraday_depth_mean"]
                                * self.data[key].spectral[:, None]
                            )
                        )
                    )
                else:
                    continue
                _ = pm.Normal(
                    key,
                    mu=stokes.sum(axis=1),
                    sigma=self.data[key].noise,
                    observed=self.data[key].brightness,
                )

            # predict Rotation measure spread function (shape: spectral, clouds)
            re_rmsf, im_rmsf = calc_rmsf(
                self.data["faraday_depth_abs"].spectral,
                self.lam2_lower,
                self.lam2_upper,
                self.num_chans,
                self.model["faraday_depth_mean"],
                self.model["polarized_intensity"],
                self.model["pol_angle0"],
            )

            # Sum over clouds to caluculate faraday depth (shape: spectral)
            faraday_depth = pt.sqrt(
                pt.sum(re_rmsf, axis=1) ** 2.0 + pt.sum(im_rmsf, axis=1) ** 2.0
            )

            _ = pm.Rice(
                "faraday_depth_abs",
                nu=faraday_depth,
                sigma=self.model["faraday_depth_sigma"],
                observed=self.data["faraday_depth_abs"].brightness,
            )
