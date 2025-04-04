"""
faraday_qu_model.py
FaradayQUModel definition

Copyright(C) 2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import pymc as pm
from pymc.distributions.transforms import CircularTransform, Ordered
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel

from bayes_pol.utils import predict_faraday_dispersion


class FaradayQUModel(BaseModel):
    """Definition of the model"""

    def __init__(
        self,
        *args,
        lam2_lower=None,
        lam2_upper=None,
        lam2_chans=None,
        **kwargs,
    ):
        """Initialize a new model instance"""
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Check data
        if len(self.data["faraday_depth_abs"].spectral) % 2 == 0:
            raise ValueError("Faraday depth spectrum length must be odd")

        # Select features used for posterior clustering
        self._cluster_features += [
            "polarization_fraction",
            "faraday_depth_mean",
            "faraday_depth_fwhm",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "polarization_fraction": r"P (\%)",
                "faraday_depth_mean": r"$\langle F \rangle$ (rad m$^{-2}$)",
                "faraday_depth_fwhm": r"$\Delta F$ (rad m$^{-2}$)",
                "pol_angle0": r"$\phi_0$ (rad)",
            }
        )

        # Get upper and lower frequency and lambda^2 limits of windows
        self.lam2_lower = np.array(lam2_lower)
        self.lam2_upper = np.array(lam2_upper)
        self.lam2_chans = np.array(lam2_chans)

    def add_priors(
        self,
        prior_faraday_depth_mean: Iterable[float] = [0.0, 100.0],  # rad m-2
        prior_faraday_depth_fwhm: float = 5.0,  # rad m-2
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_faraday_depth_mean : Iterable[float], optional
            Prior distribution on the mean Faraday depth (rad/m2), by default [0.0, 1000.0], where
            faraday_depth_mean ~ Normal(mu=prior[0], sigma=prior[1])
        prior_faraday_depth_fwhm : float, optional
            Prior distribution on the Faraday depth full-width at half-maximum (rad/m2), by default 10.0, where
            faraday_depth_fwhm ~ HalfNormal(sigma=prior)
        prior_faraday_depth_abs_sigma : float, optional
            Prior distribution on the Faraday depth magnitude uncertainty, by default 1.0, where
            faraday_depth_abs_sigma ~ HalfNormal(sigma=prior)
        """
        with self.model:
            # Polarized intensity (data brightness units)
            _ = pm.Beta("polarization_fraction", alpha=2.0, beta=2.0, dims="cloud")

            # Mean Faraday depth (rad m-2)
            faraday_depth_mean_norm = pm.Normal(
                "faraday_depth_mean_norm",
                mu=0.0,
                sigma=1.0,
                initval=np.linspace(-3.0, 3.0, self.n_clouds),
                transform=Ordered(),
                dims="cloud",
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

            # Likelihood mixture weights
            _ = pm.Dirichlet("mixture_weight", a=np.ones(2), shape=(2,))

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "Q", "U", and "faraday_depth_abs".
        Spectral units for "Q" and "U" should be square wavelength in m2.
        Spectral units for "faraday_depth_abs" should be rad/m2.
        Order of clouds is nearest to farthest.
        """
        with self.model:
            for key in self.data.keys():
                if "Q" in key:
                    func = pt.cos
                elif "U" in key:
                    func = pt.sin
                else:
                    continue

                # Predict Stokes Q and U (shape: spectral, clouds)
                stokes = (
                    self.model["polarization_fraction"]
                    * pt.exp(
                        -self.model["faraday_depth_fwhm"] ** 2.0
                        * self.data[key].spectral[:, None] ** 2.0
                        / (4.0 * np.log(2.0))
                    )
                    * func(
                        2.0
                        * (
                            self.model["pol_angle0"]
                            + self.model["faraday_depth_mean"]
                            * self.data[key].spectral[:, None]
                        )
                    )
                )

                # Sum over clouds (shape: spectral)
                _ = pm.Normal(
                    key,
                    mu=stokes.sum(axis=1),
                    sigma=self.data[key].noise,
                    observed=self.data[key].brightness,
                )

            # predict FDF (shape: spectral, clouds)
            fdf_real, fdf_imag = predict_faraday_dispersion(
                self.data["faraday_depth_abs"].spectral,
                self.model["polarization_fraction"],
                self.model["faraday_depth_mean"],
                self.model["faraday_depth_fwhm"],
                self.model["pol_angle0"],
                self.lam2_lower,
                self.lam2_upper,
                self.lam2_chans,
            )

            # Sum over clouds (shape: spectral)
            fdf_abs = pt.sqrt(fdf_real.sum(axis=1) ** 2.0 + fdf_imag.sum(axis=1) ** 2.0)

            # Mix likelihood
            components = [
                pm.Rice.dist(nu=fdf_abs, sigma=self.data["faraday_depth_abs"].noise),
                pm.TruncatedNormal.dist(
                    mu=fdf_abs, sigma=self.data["faraday_depth_abs"].noise, lower=0.0
                ),
            ]
            _ = pm.Mixture(
                "faraday_depth_abs",
                w=self.model["mixture_weight"],
                comp_dists=components,
                observed=self.data["faraday_depth_abs"].brightness,
            )
