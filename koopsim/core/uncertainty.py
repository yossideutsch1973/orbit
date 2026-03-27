"""Uncertainty quantification via Monte Carlo perturbation of initial conditions."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from koopsim.core.base import KoopmanModel
from koopsim.core.prediction import PredictionEngine


class MonteCarloUQ:
    """Monte Carlo uncertainty quantification for Koopman predictions.

    Estimates prediction uncertainty by perturbing the initial condition
    and computing statistics over the resulting ensemble of predictions.

    Parameters
    ----------
    model : KoopmanModel
        A fitted Koopman model.
    n_samples : int
        Number of Monte Carlo samples.
    noise_model : str
        Noise distribution: ``'gaussian'`` or ``'uniform'``.
    noise_scale : float
        Scale of the perturbation (std for Gaussian, half-width for uniform).
    """

    def __init__(
        self,
        model: KoopmanModel,
        n_samples: int = 100,
        noise_model: str = "gaussian",
        noise_scale: float = 0.01,
    ) -> None:
        if noise_model not in ("gaussian", "uniform"):
            raise ValueError(
                f"Unknown noise_model '{noise_model}'. Choose 'gaussian' or 'uniform'."
            )
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        if noise_scale < 0:
            raise ValueError("noise_scale must be >= 0.")

        self._model = model
        self._n_samples = n_samples
        self._noise_model = noise_model
        self._noise_scale = noise_scale
        self._engine = PredictionEngine(model)
        self._rng = np.random.default_rng(0)

    def predict_with_uncertainty(self, x0: np.ndarray, t: float) -> dict:
        """Predict system state at time t with uncertainty estimates.

        Perturbs x0 ``n_samples`` times, predicts each perturbed initial
        condition, and computes ensemble statistics.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state, shape ``(n_features,)`` or ``(1, n_features)``.
        t : float
            Target prediction time.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``mean``: ensemble mean prediction, shape ``(n_features,)``
            - ``std``: ensemble standard deviation, shape ``(n_features,)``
            - ``percentiles``: dict with keys 5, 25, 50, 75, 95,
              each shape ``(n_features,)``
            - ``samples``: all ensemble predictions, shape ``(n_samples, n_features)``
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        n_features = x0.shape[0]

        # Generate perturbed initial conditions
        predictions = np.empty((self._n_samples, n_features), dtype=np.float64)

        for i in tqdm(range(self._n_samples), desc="MC samples", leave=False):
            x0_perturbed = self._perturb(x0)
            predictions[i] = self._engine.predict(x0_perturbed, t)

        # Compute statistics
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        percentiles = {p: np.percentile(predictions, p, axis=0) for p in (5, 25, 50, 75, 95)}

        return {
            "mean": mean,
            "std": std,
            "percentiles": percentiles,
            "samples": predictions,
        }

    def _perturb(self, x0: np.ndarray) -> np.ndarray:
        """Generate a perturbed version of x0.

        Parameters
        ----------
        x0 : np.ndarray, shape (n_features,)
            Original initial condition.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Perturbed initial condition.
        """
        if self._noise_model == "gaussian":
            noise = self._rng.normal(0, self._noise_scale, size=x0.shape)
        else:  # uniform
            noise = self._rng.uniform(-self._noise_scale, self._noise_scale, size=x0.shape)
        return x0 + noise
