bayes_pol
=========

``bayes_pol`` implements spectroscopic polarization models for radio astronomy. The ``FaradayModel`` predicts Stokes Q, U, and Faraday depth spectra
assuming a Gaussian decomposition in Faraday depth space. This is a work in progress.

``bayes_pol`` is written in the ``bayes_spec`` Bayesian modeling framework, which provides methods to fit these models to data using numerical techniques
such as Monte Carlo Markov Chain (MCMC).

Useful information can be found in the `bayes_pol Github repository <https://github.com/tvwenger/bayes_pol>`_, 
the `bayes_spec Github repository <https://github.com/tvwenger/bayes_spec>`_, and in the tutorials below.

============
Installation
============
.. code-block::

    conda create --name bayes_pol -c conda-forge pymc>=5.20 pip
    conda activate bayes_pol
    pip install bayes_pol

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/faraday_model

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules
