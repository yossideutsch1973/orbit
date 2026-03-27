"""High-level public API for KoopSim.

Provides a single :class:`KoopSim` facade that wraps model fitting,
prediction, uncertainty quantification, validation, and I/O.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from koopsim.core.auto_tune import AutoTuneResult, auto_tune
from koopsim.core.base import KoopmanModel
from koopsim.core.edmd import EDMD
from koopsim.core.exceptions import KoopSimError, NotFittedError
from koopsim.core.prediction import PredictionEngine
from koopsim.core.uncertainty import MonteCarloUQ
from koopsim.core.validation import ModelValidator
from koopsim.utils.dictionary import (
    CompositeDictionary,
    IdentityDictionary,
    PolynomialDictionary,
    RBFDictionary,
)
from koopsim.utils.io import load_model, save_model

if TYPE_CHECKING:
    from koopsim.systems.base import DynamicalSystem

logger = logging.getLogger("koopsim")


class KoopSim:
    """High-level API for Koopman operator learning and prediction.

    Example::

        sim = KoopSim(method="edmd", poly_degree=3)
        sim.fit(X, Y, dt=0.01)
        x_future = sim.predict(x0, t=1.0)
        trajectory = sim.predict_trajectory(x0, np.linspace(0, 10, 100))
    """

    def __init__(
        self,
        method: str = "edmd",
        # Dictionary params (for EDMD)
        poly_degree: int | None = None,
        rbf_centers: int | None = None,
        rbf_gamma: float | str = "auto",
        regularization: float = 1e-6,
        svd_rank: int | None = None,
        # Neural params
        latent_dim: int = 16,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 64,
        # Prediction params
        prediction_method: str = "auto",
        # General
        verbose: bool = True,
    ) -> None:
        """Initialise a KoopSim instance.

        Parameters
        ----------
        method : str
            Learning method: ``"edmd"`` or ``"neural"``.
        poly_degree : int or None
            If set, include a polynomial dictionary of this degree.
        rbf_centers : int or None
            If set, include an RBF dictionary with this many centres.
        rbf_gamma : float or ``"auto"``
            Gamma parameter for RBF kernel.  ``"auto"`` uses the median heuristic.
        regularization : float
            Tikhonov regularisation for EDMD.
        svd_rank : int or None
            SVD rank truncation for EDMD.
        latent_dim : int
            Latent space dimension for the neural method.
        encoder_hidden : list[int] or None
            Hidden layer sizes for the neural encoder.
        decoder_hidden : list[int] or None
            Hidden layer sizes for the neural decoder.
        lr : float
            Learning rate for the neural method.
        max_epochs : int
            Maximum training epochs for the neural method.
        batch_size : int
            Batch size for the neural method.
        prediction_method : str
            Prediction backend: ``"auto"``, ``"expm"``, or ``"eigen"``.
        verbose : bool
            If ``True``, print progress information.
        """
        self.method = method
        self.prediction_method = prediction_method
        self.verbose = verbose

        # Store all params for deferred model building and repr
        self._poly_degree = poly_degree
        self._rbf_centers = rbf_centers
        self._rbf_gamma = rbf_gamma
        self._regularization = regularization
        self._svd_rank = svd_rank
        self._latent_dim = latent_dim
        self._encoder_hidden = encoder_hidden
        self._decoder_hidden = decoder_hidden
        self._lr = lr
        self._max_epochs = max_epochs
        self._batch_size = batch_size

        # Build the model (unfitted)
        self._model: KoopmanModel | None = None
        self._engine: PredictionEngine | None = None
        self._build_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        """Construct the underlying Koopman model (unfitted)."""
        if self.method == "edmd":
            dictionaries: list = []
            if self._poly_degree is not None:
                dictionaries.append(PolynomialDictionary(self._poly_degree))
            if self._rbf_centers is not None:
                dictionaries.append(
                    RBFDictionary(n_centers=self._rbf_centers, gamma=self._rbf_gamma)
                )
            if dictionaries:
                dictionary = CompositeDictionary(dictionaries)
            else:
                dictionary = IdentityDictionary()
            self._model = EDMD(
                dictionary=dictionary,
                regularization=self._regularization,
                svd_rank=self._svd_rank,
            )
        elif self.method == "neural":
            from koopsim.core.neural_koopman import NeuralKoopman  # type: ignore[import-not-found]

            self._model = NeuralKoopman(
                latent_dim=self._latent_dim,
                encoder_hidden=self._encoder_hidden,
                decoder_hidden=self._decoder_hidden,
                lr=self._lr,
                max_epochs=self._max_epochs,
                batch_size=self._batch_size,
            )
        else:
            raise KoopSimError(f"Unknown method '{self.method}'. Choose 'edmd' or 'neural'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, dt: float) -> KoopSim:
        """Fit the Koopman model on snapshot pairs.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Pre-snapshot data (row vectors).
        Y : np.ndarray, shape (n_samples, n_features)
            Post-snapshot data (row vectors), taken *dt* time later.
        dt : float
            Time step between snapshot pairs.

        Returns
        -------
        self
        """
        if self._model is None:
            raise KoopSimError("Model not initialised.")
        self._model.fit(X, Y, dt)
        self._engine = PredictionEngine(self._model, method=self.prediction_method)
        if self.verbose:
            logger.info(
                "KoopSim fitted (%s): %d samples, %d state dims -> %d Koopman dims.",
                self.method,
                X.shape[0],
                self._model.n_state_dims,
                self._model.n_koopman_dims,
            )
        return self

    def fit_auto(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        dt: float,
        *,
        poly_degrees: list[int] | None = None,
        rbf_centers_list: list[int] | None = None,
        regularizations: list[float] | None = None,
        n_folds: int = 5,
        metric: str = "rmse",
    ) -> AutoTuneResult:
        """Auto-select hyperparameters via cross-validation, then fit.

        Evaluates combinations of dictionary type/complexity and
        regularization using k-fold CV, selects the best, and fits
        the final model on all data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Pre-snapshot data.
        Y : np.ndarray, shape (n_samples, n_features)
            Post-snapshot data.
        dt : float
            Time step between snapshot pairs.
        poly_degrees : list[int] or None
            Polynomial degrees to try. Default: [2, 3, 4].
        rbf_centers_list : list[int] or None
            RBF center counts to try. Default: [10, 25, 50].
        regularizations : list[float] or None
            Regularization values to try. Default: [1e-8, 1e-6, 1e-4, 1e-2].
        n_folds : int
            Number of CV folds.
        metric : str
            Error metric for CV: 'rmse', 'mae', or 'relative'.

        Returns
        -------
        AutoTuneResult
            The best hyperparameters and all CV results.
        """
        result = auto_tune(
            X,
            Y,
            dt,
            poly_degrees=poly_degrees,
            rbf_centers_list=rbf_centers_list,
            regularizations=regularizations,
            n_folds=n_folds,
            metric=metric,
            verbose=self.verbose,
        )

        # Apply best hyperparameters and refit on all data
        self._poly_degree = result.poly_degree
        self._rbf_centers = result.rbf_centers
        self._regularization = result.regularization
        self.method = "edmd"
        self._build_model()
        self.fit(X, Y, dt)

        return result

    def predict(self, x0: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Predict state at time *t* (scalar or array).

        Parameters
        ----------
        x0 : np.ndarray
            Initial state, shape ``(n_state,)`` or ``(n_samples, n_state)``.
        t : float or np.ndarray
            Single time or 1-D array of query times.

        Returns
        -------
        np.ndarray
            Predicted state(s).

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        """
        if self._engine is None:
            raise NotFittedError("Call fit() before predict().")
        return self._engine.predict(x0, t)

    def predict_trajectory(self, x0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict trajectory at specified times.

        Parameters
        ----------
        x0 : np.ndarray, shape ``(n_state,)``
            Initial state vector.
        times : np.ndarray, shape ``(n_times,)``
            Array of query times.

        Returns
        -------
        np.ndarray, shape ``(n_times, n_state)``

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        """
        if self._engine is None:
            raise NotFittedError("Call fit() before predict_trajectory().")
        return self._engine.predict_trajectory(x0, times)

    def predict_with_uncertainty(
        self,
        x0: np.ndarray,
        t: float,
        n_samples: int = 100,
        noise_scale: float = 0.01,
        noise_model: str = "gaussian",
    ) -> dict:
        """Predict with Monte Carlo uncertainty quantification.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state vector.
        t : float
            Target prediction time.
        n_samples : int
            Number of Monte Carlo samples.
        noise_scale : float
            Scale of perturbation applied to *x0*.
        noise_model : str
            ``"gaussian"`` or ``"uniform"``.

        Returns
        -------
        dict
            Keys: ``mean``, ``std``, ``percentiles``, ``samples``.
        """
        if self._model is None or not self._model._is_fitted():
            raise NotFittedError("Call fit() before predict_with_uncertainty().")
        uq = MonteCarloUQ(
            self._model,
            n_samples=n_samples,
            noise_model=noise_model,
            noise_scale=noise_scale,
        )
        return uq.predict_with_uncertainty(x0, t)

    def validate(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        metric: str = "rmse",
    ) -> float:
        """Compute prediction error on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Pre-snapshot test data.
        Y_test : np.ndarray
            Post-snapshot test data (ground truth).
        metric : str
            ``"rmse"``, ``"mae"``, or ``"relative"``.

        Returns
        -------
        float
        """
        if self._model is None or not self._model._is_fitted():
            raise NotFittedError("Call fit() before validate().")
        return ModelValidator.prediction_error(self._model, X_test, Y_test, metric=metric)

    def spectral_analysis(self) -> dict:
        """Analyse eigenvalues of the Koopman matrix.

        Returns
        -------
        dict
            Keys: ``eigenvalues``, ``frequencies``, ``growth_rates``,
            ``dominant_mode_indices``.
        """
        if self._model is None or not self._model._is_fitted():
            raise NotFittedError("Call fit() before spectral_analysis().")
        return ModelValidator.spectral_analysis(self._model)

    def save(self, path: str) -> None:
        """Save fitted model to a ``.koop`` file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        if self._model is None or not self._model._is_fitted():
            raise NotFittedError("Call fit() before save().")
        save_model(self._model, path)

    @classmethod
    def load(cls, path: str, prediction_method: str = "auto") -> KoopSim:
        """Load a model from a ``.koop`` file.

        Parameters
        ----------
        path : str
            Path to the ``.koop`` file.
        prediction_method : str
            Prediction backend for the loaded model.

        Returns
        -------
        KoopSim
        """
        model = load_model(path)
        instance = cls.__new__(cls)
        instance._model = model
        # Infer method from model type
        instance.method = "edmd" if isinstance(model, EDMD) else "neural"
        instance.prediction_method = prediction_method
        instance.verbose = True
        instance._engine = PredictionEngine(model, method=prediction_method)
        # Set parameter attributes to defaults so repr/access doesn't fail
        instance._poly_degree = None
        instance._rbf_centers = None
        instance._rbf_gamma = "auto"
        instance._regularization = 1e-6
        instance._svd_rank = None
        instance._latent_dim = 16
        instance._encoder_hidden = None
        instance._decoder_hidden = None
        instance._lr = 1e-3
        instance._max_epochs = 100
        instance._batch_size = 64
        return instance

    @classmethod
    def from_system(
        cls,
        system: DynamicalSystem,
        x0: np.ndarray | None = None,
        dt: float = 0.01,
        n_steps: int = 100,
        n_trajectories: int = 10,
        noise_std: float = 0.0,
        **kwargs,
    ) -> KoopSim:
        """Generate data from a DynamicalSystem, fit, and return.

        Convenience factory that creates snapshot pairs from the given system,
        constructs a :class:`KoopSim` instance with the provided ``**kwargs``,
        and fits it.

        Parameters
        ----------
        system : DynamicalSystem
            System to generate training data from.
        x0 : np.ndarray or None
            Initial condition.  If ``None``, a small random initial condition
            is used.
        dt : float
            Time step between snapshots.
        n_steps : int
            Number of steps per trajectory.
        n_trajectories : int
            Number of trajectories to generate.
        noise_std : float
            Additive noise standard deviation.
        **kwargs
            Passed to the :class:`KoopSim` constructor.

        Returns
        -------
        KoopSim
            A fitted instance.
        """
        if x0 is None:
            rng = np.random.default_rng(42)
            x0 = rng.standard_normal(system.state_dim) * 0.5
        X, Y = system.generate_snapshots(x0, dt, n_steps, n_trajectories, noise_std)
        instance = cls(**kwargs)
        instance.fit(X, Y, dt)
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> KoopmanModel:
        """The underlying Koopman model."""
        if self._model is None:
            raise KoopSimError("Model not initialised.")
        return self._model

    def __repr__(self) -> str:
        fitted = self._model._is_fitted() if self._model is not None else False
        return (
            f"KoopSim(method={self.method!r}, fitted={fitted}, "
            f"prediction_method={self.prediction_method!r})"
        )
