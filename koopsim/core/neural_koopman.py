"""Neural Koopman Autoencoder implementation.

Learns a Koopman operator via an autoencoder architecture:
encoder MLP -> linear Koopman layer (no bias) -> decoder MLP.

Requires the ``neural`` extra: ``pip install koopsim[neural]``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from koopsim.core.base import KoopmanModel
from koopsim.core.exceptions import NotFittedError

logger = logging.getLogger("koopsim")


def _import_torch():
    """Lazy import of torch and lightning with a clear error message."""
    try:
        import lightning as pl
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "Neural Koopman requires PyTorch and Lightning. "
            "Install them with: pip install koopsim[neural]"
        ) from exc
    return torch, nn, F, pl


def _build_mlp(
    layer_sizes: list[int],
    nn_module: Any,
) -> Any:
    """Build a simple MLP from a list of layer sizes.

    Uses ReLU activations between layers; no activation on the final layer.
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn_module.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn_module.ReLU())
    return nn_module.Sequential(*layers)


class KoopmanAutoencoder:
    """Autoencoder with linear Koopman layer.

    Architecture: encoder MLP -> K linear layer (no bias) -> decoder MLP.

    This class is created via the factory :func:`_create_autoencoder` to allow
    lazy torch imports.  It inherits from ``pl.LightningModule`` at runtime.

    Parameters
    ----------
    state_dim : int
        Dimension of the original state space.
    latent_dim : int
        Dimension of the Koopman latent space.
    encoder_hidden : list[int]
        Hidden layer sizes for the encoder MLP.
    decoder_hidden : list[int]
        Hidden layer sizes for the decoder MLP.
    lr : float
        Learning rate for Adam optimizer.
    loss_weights : dict
        Weights for the three loss terms: ``reconstruction``, ``prediction``,
        and ``linearity``.
    """

    # Sentinel — the real class is built dynamically in _create_autoencoder.
    pass


def _create_autoencoder(
    state_dim: int,
    latent_dim: int,
    encoder_hidden: list[int],
    decoder_hidden: list[int],
    lr: float,
    loss_weights: dict,
):
    """Create a KoopmanAutoencoder instance backed by pl.LightningModule.

    This factory avoids top-level torch/lightning imports.
    """
    torch, nn, F, pl = _import_torch()

    class _KoopmanAutoencoder(pl.LightningModule):
        """Autoencoder with linear Koopman layer (runtime class)."""

        def __init__(self):
            super().__init__()
            self.lr = lr
            self.loss_weights = dict(loss_weights)

            # Encoder: state_dim -> hidden... -> latent_dim
            enc_sizes = [state_dim] + list(encoder_hidden) + [latent_dim]
            self.encoder = _build_mlp(enc_sizes, nn)

            # Koopman linear layer (no bias)
            self.K_layer = nn.Linear(latent_dim, latent_dim, bias=False)

            # Decoder: latent_dim -> hidden... -> state_dim
            dec_sizes = [latent_dim] + list(decoder_hidden) + [state_dim]
            self.decoder = _build_mlp(dec_sizes, nn)

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            """Returns (x_reconstructed, y_predicted, z_x, z_next)."""
            z = self.encode(x)
            x_recon = self.decode(z)
            z_next = self.K_layer(z)
            y_pred = self.decode(z_next)
            return x_recon, y_pred, z, z_next

        def training_step(self, batch, batch_idx):
            x, y = batch
            x_recon, y_pred, z_x, z_y_pred = self(x)
            z_y = self.encode(y)

            # Three-term loss
            loss_recon = F.mse_loss(x_recon, x)
            loss_pred = F.mse_loss(y_pred, y)
            loss_linear = F.mse_loss(z_y_pred, z_y)

            loss = (
                self.loss_weights["reconstruction"] * loss_recon
                + self.loss_weights["prediction"] * loss_pred
                + self.loss_weights["linearity"] * loss_linear
            )
            self.log_dict(
                {
                    "loss": loss,
                    "recon": loss_recon,
                    "pred": loss_pred,
                    "linear": loss_linear,
                },
                prog_bar=True,
            )
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    return _KoopmanAutoencoder()


class NeuralKoopman(KoopmanModel):
    """KoopmanModel wrapper for the neural autoencoder approach.

    Handles PyTorch training internally via :meth:`fit`.

    Parameters
    ----------
    latent_dim : int
        Dimension of the Koopman latent space.
    encoder_hidden : list[int]
        Hidden layer sizes for the encoder MLP.
    decoder_hidden : list[int]
        Hidden layer sizes for the decoder MLP.
    lr : float
        Learning rate for Adam optimizer.
    max_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size for training.
    loss_weights : dict or None
        Weights for the three loss terms.  Defaults to
        ``{"reconstruction": 1.0, "prediction": 1.0, "linearity": 0.1}``.
    verbose : bool
        If ``True``, show a progress bar during training.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 64,
        loss_weights: dict | None = None,
        verbose: bool = True,
    ) -> None:
        # Verify torch/lightning are available early
        _import_torch()

        self._latent_dim = latent_dim
        self._encoder_hidden = encoder_hidden if encoder_hidden is not None else [64, 64]
        self._decoder_hidden = decoder_hidden if decoder_hidden is not None else [64, 64]
        self._lr = lr
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._loss_weights = loss_weights or {
            "reconstruction": 1.0,
            "prediction": 1.0,
            "linearity": 0.1,
        }
        self._verbose = verbose

        # Fitted attributes (set by fit)
        self.K_: np.ndarray | None = None
        self._autoencoder = None
        self._n_state_dims: int | None = None
        self._dt: float | None = None

    # ------------------------------------------------------------------
    # KoopmanModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, dt: float) -> NeuralKoopman:
        """Train the autoencoder on snapshot pairs.

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
        torch, nn, F, pl = _import_torch()

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        if X.ndim != 2 or Y.ndim != 2:
            from koopsim.core.exceptions import DimensionMismatchError

            raise DimensionMismatchError(
                f"X and Y must be 2D arrays. Got X.ndim={X.ndim}, Y.ndim={Y.ndim}."
            )
        if X.shape != Y.shape:
            from koopsim.core.exceptions import DimensionMismatchError

            raise DimensionMismatchError(
                f"X and Y must have the same shape. Got X.shape={X.shape}, Y.shape={Y.shape}."
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")

        n_samples, state_dim = X.shape
        self._n_state_dims = state_dim
        self._dt = dt

        logger.info(
            "NeuralKoopman fitting: %d samples, %d state dims -> %d latent dims.",
            n_samples,
            state_dim,
            self._latent_dim,
        )

        # Create dataset and dataloader
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
        )

        # Build autoencoder
        autoencoder = _create_autoencoder(
            state_dim=state_dim,
            latent_dim=self._latent_dim,
            encoder_hidden=self._encoder_hidden,
            decoder_hidden=self._decoder_hidden,
            lr=self._lr,
            loss_weights=self._loss_weights,
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=self._max_epochs,
            enable_progress_bar=self._verbose,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(autoencoder, dataloader)

        # Store trained autoencoder
        self._autoencoder = autoencoder
        self._autoencoder.eval()

        # Extract K matrix.
        # nn.Linear computes output = input @ weight.T (+ bias).
        # Our row-vector convention: z_next = z @ K, so K = weight.T.
        weight = autoencoder.K_layer.weight.detach().cpu().numpy()
        self.K_ = weight.T.astype(np.float64)

        logger.info("NeuralKoopman fit complete. K shape: %s.", self.K_.shape)
        return self

    def get_koopman_matrix(self) -> np.ndarray:
        """Return the fitted Koopman matrix K.

        Returns
        -------
        np.ndarray, shape (latent_dim, latent_dim)

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self.K_ is None:
            raise NotFittedError("NeuralKoopman has not been fitted. Call fit() first.")
        return self.K_

    def lift(self, X: np.ndarray) -> np.ndarray:
        """Encode state-space data into the Koopman latent space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        np.ndarray, shape (n_samples, latent_dim) or (latent_dim,)

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self._autoencoder is None:
            raise NotFittedError("NeuralKoopman has not been fitted. Call fit() first.")

        torch, nn, F, pl = _import_torch()

        X = np.asarray(X, dtype=np.float32)
        squeeze = X.ndim == 1
        if squeeze:
            X = X.reshape(1, -1)

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            Z_t = self._autoencoder.encode(X_t)
            Z = Z_t.cpu().numpy().astype(np.float64)

        if squeeze:
            return Z[0]
        return Z

    def unlift(self, Z: np.ndarray) -> np.ndarray:
        """Decode Koopman latent vectors back to state space.

        Parameters
        ----------
        Z : np.ndarray, shape (n_samples, latent_dim) or (latent_dim,)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features) or (n_features,)

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self._autoencoder is None:
            raise NotFittedError("NeuralKoopman has not been fitted. Call fit() first.")

        torch, nn, F, pl = _import_torch()

        Z = np.asarray(Z, dtype=np.float32)
        squeeze = Z.ndim == 1
        if squeeze:
            Z = Z.reshape(1, -1)

        with torch.no_grad():
            Z_t = torch.tensor(Z, dtype=torch.float32)
            X_t = self._autoencoder.decode(Z_t)
            X = X_t.cpu().numpy().astype(np.float64)

        if squeeze:
            return X[0]
        return X

    @property
    def n_state_dims(self) -> int:
        """Number of original state-space dimensions."""
        if self._n_state_dims is None:
            raise NotFittedError("NeuralKoopman has not been fitted. Call fit() first.")
        return self._n_state_dims

    @property
    def n_koopman_dims(self) -> int:
        """Number of Koopman latent dimensions."""
        if self.K_ is None:
            raise NotFittedError("NeuralKoopman has not been fitted. Call fit() first.")
        return self._latent_dim

    @property
    def dt(self) -> float:
        """Time step between snapshot pairs used for fitting."""
        if self._dt is None:
            raise NotFittedError("NeuralKoopman has not been fitted. Call fit() first.")
        return self._dt
