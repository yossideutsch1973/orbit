"""Tests for model I/O (save/load .koop files)."""

from __future__ import annotations

import numpy as np
import pytest
import h5py

from koopsim.core.edmd import EDMD
from koopsim.utils.dictionary import (
    CompositeDictionary,
    IdentityDictionary,
    PolynomialDictionary,
    RBFDictionary,
)
from koopsim.utils.io import save_model, load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_identity_model(simple_linear_system):
    """EDMD with IdentityDictionary fitted on the rotation system."""
    X, Y, dt, _ = simple_linear_system
    model = EDMD(dictionary=IdentityDictionary(), regularization=1e-15)
    model.fit(X, Y, dt)
    return model


@pytest.fixture
def fitted_composite_model(rng):
    """EDMD with CompositeDictionary (Poly + RBF) fitted on rotation system."""
    theta = np.pi / 6
    dt = 0.1
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    n_samples = 200
    X = rng.standard_normal((n_samples, 2))
    Y = X @ R.T
    poly = PolynomialDictionary(degree=3)
    rbf = RBFDictionary(n_centers=10, gamma=0.5)
    dictionary = CompositeDictionary([poly, rbf])
    model = EDMD(dictionary=dictionary, regularization=1e-10)
    model.fit(X, Y, dt)
    return model, X, Y


# ---------------------------------------------------------------------------
# Save/load roundtrip with Identity dictionary
# ---------------------------------------------------------------------------


class TestIdentitySaveLoad:
    """Save/load roundtrip with EDMD + IdentityDictionary."""

    def test_roundtrip_identity(self, fitted_identity_model, tmp_path):
        """Save and load should produce an equivalent model."""
        model = fitted_identity_model
        path = tmp_path / "model_identity.koop"

        save_model(model, path)
        loaded = load_model(path)

        assert isinstance(loaded, EDMD)
        np.testing.assert_array_equal(loaded.K_, model.K_)
        assert loaded.n_state_dims == model.n_state_dims
        assert loaded.n_koopman_dims == model.n_koopman_dims
        assert loaded.dt == model.dt

    def test_loaded_produces_same_predictions(
        self, fitted_identity_model, simple_linear_system, tmp_path
    ):
        """Loaded model should produce the same one-step predictions."""
        model = fitted_identity_model
        X, Y, dt, _ = simple_linear_system
        path = tmp_path / "model_pred.koop"

        save_model(model, path)
        loaded = load_model(path)

        K_orig = model.get_koopman_matrix()
        K_loaded = loaded.get_koopman_matrix()

        Y_pred_orig = model.unlift(model.lift(X) @ K_orig)
        Y_pred_loaded = loaded.unlift(loaded.lift(X) @ K_loaded)

        np.testing.assert_allclose(Y_pred_loaded, Y_pred_orig, atol=1e-14)

    def test_file_is_valid_hdf5(self, fitted_identity_model, tmp_path):
        """Saved file should be a valid HDF5 file."""
        model = fitted_identity_model
        path = tmp_path / "model_hdf5.koop"

        save_model(model, path)

        # h5py should be able to open and read it
        with h5py.File(path, "r") as f:
            assert "K" in f
            assert "model_class" in f.attrs
            assert f.attrs["model_class"] == "EDMD"


# ---------------------------------------------------------------------------
# Save/load roundtrip with Composite dictionary (Poly + RBF)
# ---------------------------------------------------------------------------


class TestCompositeSaveLoad:
    """Save/load roundtrip with EDMD + CompositeDictionary (Poly + RBF)."""

    def test_roundtrip_composite(self, fitted_composite_model, tmp_path):
        """Save and load should produce an equivalent model."""
        model, X, Y = fitted_composite_model
        path = tmp_path / "model_composite.koop"

        save_model(model, path)
        loaded = load_model(path)

        assert isinstance(loaded, EDMD)
        np.testing.assert_allclose(loaded.K_, model.K_, atol=1e-14)
        assert loaded.n_state_dims == model.n_state_dims
        assert loaded.n_koopman_dims == model.n_koopman_dims
        assert loaded.dt == model.dt

    def test_loaded_composite_same_predictions(
        self, fitted_composite_model, tmp_path
    ):
        """Loaded composite model should produce the same predictions."""
        model, X, Y = fitted_composite_model
        path = tmp_path / "model_composite_pred.koop"

        save_model(model, path)
        loaded = load_model(path)

        K_orig = model.get_koopman_matrix()
        K_loaded = loaded.get_koopman_matrix()

        Y_pred_orig = model.unlift(model.lift(X) @ K_orig)
        Y_pred_loaded = loaded.unlift(loaded.lift(X) @ K_loaded)

        np.testing.assert_allclose(Y_pred_loaded, Y_pred_orig, atol=1e-10)

    def test_composite_dictionary_lift_matches(
        self, fitted_composite_model, tmp_path
    ):
        """Loaded model's lift should produce the same lifted representation."""
        model, X, _ = fitted_composite_model
        path = tmp_path / "model_composite_lift.koop"

        save_model(model, path)
        loaded = load_model(path)

        Z_orig = model.lift(X)
        Z_loaded = loaded.lift(X)

        np.testing.assert_allclose(Z_loaded, Z_orig, atol=1e-10)

    def test_file_is_valid_hdf5(self, fitted_composite_model, tmp_path):
        """Saved composite model file should be valid HDF5."""
        model, _, _ = fitted_composite_model
        path = tmp_path / "model_composite_hdf5.koop"

        save_model(model, path)

        with h5py.File(path, "r") as f:
            assert "K" in f
            assert "dictionary" in f
            assert f.attrs["model_class"] == "EDMD"


# ---------------------------------------------------------------------------
# Save/load roundtrip with NeuralKoopman
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


@pytest.mark.neural
class TestNeuralSaveLoad:
    """Save/load roundtrip with NeuralKoopman."""

    @pytest.fixture
    def fitted_neural_model(self, simple_linear_system):
        from koopsim.core.neural_koopman import NeuralKoopman

        X, Y, dt, _ = simple_linear_system
        model = NeuralKoopman(
            latent_dim=4,
            encoder_hidden=[16, 16],
            decoder_hidden=[16, 16],
            lr=1e-3,
            max_epochs=10,
            batch_size=64,
            verbose=False,
        )
        model.fit(X, Y, dt)
        return model, X

    def test_roundtrip_neural(self, fitted_neural_model, tmp_path):
        """Save and load should produce an equivalent model."""
        model, X = fitted_neural_model
        path = tmp_path / "model_neural.koop"

        save_model(model, path)
        loaded = load_model(path)

        from koopsim.core.neural_koopman import NeuralKoopman

        assert isinstance(loaded, NeuralKoopman)
        np.testing.assert_array_equal(loaded.K_, model.K_)
        assert loaded.n_state_dims == model.n_state_dims
        assert loaded.n_koopman_dims == model.n_koopman_dims
        assert loaded.dt == model.dt

    def test_loaded_neural_same_predictions(self, fitted_neural_model, tmp_path):
        """Loaded neural model should produce the same lift/unlift results."""
        model, X = fitted_neural_model
        path = tmp_path / "model_neural_pred.koop"

        save_model(model, path)
        loaded = load_model(path)

        Z_orig = model.lift(X[:10])
        Z_loaded = loaded.lift(X[:10])
        np.testing.assert_allclose(Z_loaded, Z_orig, atol=1e-6)

        X_rec_orig = model.unlift(Z_orig)
        X_rec_loaded = loaded.unlift(Z_loaded)
        np.testing.assert_allclose(X_rec_loaded, X_rec_orig, atol=1e-6)

    def test_file_is_valid_hdf5(self, fitted_neural_model, tmp_path):
        """Saved neural model file should be valid HDF5."""
        model, _ = fitted_neural_model
        path = tmp_path / "model_neural_hdf5.koop"

        save_model(model, path)

        with h5py.File(path, "r") as f:
            assert "K" in f
            assert "autoencoder_state_dict" in f
            assert f.attrs["model_class"] == "NeuralKoopman"
