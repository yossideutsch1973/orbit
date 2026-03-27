"""Model I/O for saving and loading Koopman models as .koop (HDF5) files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np

from koopsim.core.base import KoopmanModel
from koopsim.core.exceptions import KoopSimError

logger = logging.getLogger("koopsim")


def save_model(model: KoopmanModel, path: str | Path) -> None:
    """Save a fitted KoopmanModel to a .koop file (HDF5).

    Parameters
    ----------
    model : KoopmanModel
        A fitted Koopman model to save.
    path : str or Path
        Destination file path. Will be created or overwritten.

    Raises
    ------
    KoopSimError
        If the model is not fitted or the model type is unsupported.
    """
    from koopsim.core.edmd import EDMD
    from koopsim.core.neural_koopman import NeuralKoopman

    path = Path(path)

    if not model._is_fitted():
        raise KoopSimError("Cannot save an unfitted model.")

    if isinstance(model, NeuralKoopman):
        _save_neural_model(model, path)
    elif isinstance(model, EDMD):
        _save_edmd_model(model, path)
    else:
        raise KoopSimError(f"Unsupported model type for saving: {type(model).__name__}.")


def _save_edmd_model(model, path: Path) -> None:
    """Save an EDMD model to HDF5."""
    logger.info("Saving EDMD model to %s", path)

    with h5py.File(path, "w") as f:
        f.create_dataset("K", data=model.K_)

        f.attrs["model_class"] = "EDMD"
        f.attrs["n_state_dims"] = model.n_state_dims
        f.attrs["n_koopman_dims"] = model.n_koopman_dims
        f.attrs["dt"] = model.dt
        f.attrs["regularization"] = model._regularization
        if model._svd_rank is not None:
            f.attrs["svd_rank"] = model._svd_rank

        dict_config = _serialize_dictionary(model.dictionary_)
        f.attrs["dictionary_config"] = json.dumps(dict_config)

        _save_dictionary_data(f, model.dictionary_)

    logger.info("Model saved successfully to %s", path)


def _save_neural_model(model, path: Path) -> None:
    """Save a NeuralKoopman model to HDF5 + PyTorch checkpoint."""
    try:
        import torch
    except ImportError as exc:
        raise KoopSimError(
            "Saving NeuralKoopman requires PyTorch. Install it with: pip install koopsim[neural]"
        ) from exc

    logger.info("Saving NeuralKoopman model to %s", path)

    with h5py.File(path, "w") as f:
        f.create_dataset("K", data=model.K_)

        f.attrs["model_class"] = "NeuralKoopman"
        f.attrs["n_state_dims"] = model.n_state_dims
        f.attrs["n_koopman_dims"] = model.n_koopman_dims
        f.attrs["dt"] = model.dt

        # Store hyperparameters needed to reconstruct the model
        f.attrs["latent_dim"] = model._latent_dim
        f.attrs["encoder_hidden"] = json.dumps(model._encoder_hidden)
        f.attrs["decoder_hidden"] = json.dumps(model._decoder_hidden)
        f.attrs["lr"] = model._lr
        f.attrs["max_epochs"] = model._max_epochs
        f.attrs["batch_size"] = model._batch_size
        f.attrs["loss_weights"] = json.dumps(model._loss_weights)

        # Save autoencoder weights as a binary blob in HDF5
        import io as io_module

        buffer = io_module.BytesIO()
        torch.save(model._autoencoder.state_dict(), buffer)
        f.create_dataset("autoencoder_state_dict", data=np.void(buffer.getvalue()))

    logger.info("NeuralKoopman model saved successfully to %s", path)


def load_model(path: str | Path) -> KoopmanModel:
    """Load a KoopmanModel from a .koop file.

    Parameters
    ----------
    path : str or Path
        Path to the .koop file.

    Returns
    -------
    KoopmanModel
        The reconstructed fitted model.

    Raises
    ------
    KoopSimError
        If the file is invalid or the model type is unsupported.
    """
    path = Path(path)

    if not path.exists():
        raise KoopSimError(f"File not found: {path}")

    logger.info("Loading model from %s", path)

    try:
        with h5py.File(path, "r") as f:
            model_class = f.attrs["model_class"]

            if model_class == "NeuralKoopman":
                model = _load_neural_model(f)
            elif model_class == "EDMD":
                model = _load_edmd_model(f)
            else:
                raise KoopSimError(f"Unsupported model class in file: {model_class}.")
    except KoopSimError:
        raise
    except (OSError, KeyError) as exc:
        raise KoopSimError(f"Failed to load model from {path}: {exc}") from exc

    logger.info("Model loaded successfully from %s", path)
    return model


def _load_edmd_model(f: h5py.File):
    """Load an EDMD model from an open HDF5 file."""
    from koopsim.core.edmd import EDMD

    K = np.array(f["K"])
    n_state_dims = int(f.attrs["n_state_dims"])
    n_koopman_dims = int(f.attrs["n_koopman_dims"])
    dt = float(f.attrs["dt"])
    regularization = float(f.attrs["regularization"])
    svd_rank = int(f.attrs["svd_rank"]) if "svd_rank" in f.attrs else None

    dict_config = json.loads(f.attrs["dictionary_config"])
    dictionary = _deserialize_dictionary(dict_config, f)

    model = EDMD(
        dictionary=dictionary,
        regularization=regularization,
        svd_rank=svd_rank,
    )

    model.K_ = K
    model.dictionary_ = dictionary
    model._n_state_dims = n_state_dims
    model._n_koopman_dims = n_koopman_dims
    model._dt = dt

    return model


def _load_neural_model(f: h5py.File):
    """Load a NeuralKoopman model from an open HDF5 file."""
    try:
        import torch
    except ImportError as exc:
        raise KoopSimError(
            "Loading NeuralKoopman requires PyTorch. Install it with: pip install koopsim[neural]"
        ) from exc

    from koopsim.core.neural_koopman import NeuralKoopman, _create_autoencoder

    K = np.array(f["K"])
    n_state_dims = int(f.attrs["n_state_dims"])
    dt = float(f.attrs["dt"])
    latent_dim = int(f.attrs["latent_dim"])
    encoder_hidden = json.loads(f.attrs["encoder_hidden"])
    decoder_hidden = json.loads(f.attrs["decoder_hidden"])
    lr = float(f.attrs["lr"])
    max_epochs = int(f.attrs["max_epochs"])
    batch_size = int(f.attrs["batch_size"])
    loss_weights = json.loads(f.attrs["loss_weights"])

    model = NeuralKoopman(
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        lr=lr,
        max_epochs=max_epochs,
        batch_size=batch_size,
        loss_weights=loss_weights,
    )

    # Recreate autoencoder architecture and load weights
    autoencoder = _create_autoencoder(
        state_dim=n_state_dims,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        lr=lr,
        loss_weights=loss_weights,
    )

    import io as io_module

    state_dict_bytes = bytes(f["autoencoder_state_dict"][()].tobytes())
    buffer = io_module.BytesIO(state_dict_bytes)
    state_dict = torch.load(buffer, weights_only=True)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    model._autoencoder = autoencoder
    model.K_ = K
    model._n_state_dims = n_state_dims
    model._dt = dt

    return model


def _serialize_dictionary(dictionary) -> dict:
    """Serialize an ObservableDictionary to a JSON-compatible dict.

    Parameters
    ----------
    dictionary : ObservableDictionary
        The dictionary to serialize.

    Returns
    -------
    dict
        Configuration dictionary with type and parameters.
    """
    from koopsim.utils.dictionary import (
        CompositeDictionary,
        IdentityDictionary,
        PolynomialDictionary,
        RBFDictionary,
    )

    if isinstance(dictionary, CompositeDictionary):
        # Serialize each sub-dictionary (skip the auto-prepended identity at index 0)
        sub_configs = []
        for d in dictionary._dictionaries:
            sub_configs.append(_serialize_dictionary(d))
        return {"type": "CompositeDictionary", "dictionaries": sub_configs}
    elif isinstance(dictionary, IdentityDictionary):
        return {
            "type": "IdentityDictionary",
            "n_features": dictionary._n_features,
        }
    elif isinstance(dictionary, PolynomialDictionary):
        return {
            "type": "PolynomialDictionary",
            "degree": dictionary._degree,
            "n_features": dictionary._n_features,
        }
    elif isinstance(dictionary, RBFDictionary):
        return {
            "type": "RBFDictionary",
            "n_centers": dictionary._n_centers,
            "kernel": dictionary._kernel,
            "gamma": dictionary._gamma,
        }
    else:
        raise KoopSimError(f"Cannot serialize dictionary type: {type(dictionary).__name__}")


def _save_dictionary_data(f: h5py.File, dictionary) -> None:
    """Save dictionary-specific data (centers, fitted state) into the HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file in write mode.
    dictionary : ObservableDictionary
        The fitted dictionary whose data to save.
    """
    from koopsim.utils.dictionary import (
        CompositeDictionary,
    )

    if isinstance(dictionary, CompositeDictionary):
        dict_grp = f.create_group("dictionary")
        for i, d in enumerate(dictionary._dictionaries):
            sub_grp = dict_grp.create_group(f"sub_{i}")
            _save_sub_dictionary_data(sub_grp, d)
    else:
        dict_grp = f.create_group("dictionary")
        _save_sub_dictionary_data(dict_grp, dictionary)


def _save_sub_dictionary_data(grp: h5py.Group, dictionary) -> None:
    """Save data for a single (non-composite) dictionary into an HDF5 group.

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group to write into.
    dictionary : ObservableDictionary
        The dictionary whose data to save.
    """
    from koopsim.utils.dictionary import (
        RBFDictionary,
    )

    grp.attrs["type"] = type(dictionary).__name__

    if isinstance(dictionary, RBFDictionary):
        if dictionary._centers is not None:
            grp.create_dataset("centers", data=dictionary._centers)
        if dictionary._gamma is not None:
            grp.attrs["gamma"] = dictionary._gamma


def _deserialize_dictionary(config: dict, f: h5py.File):
    """Reconstruct an ObservableDictionary from its serialized config and HDF5 data.

    Parameters
    ----------
    config : dict
        Configuration dictionary from serialization.
    f : h5py.File
        Open HDF5 file to read dictionary data from.

    Returns
    -------
    ObservableDictionary
        Reconstructed and fitted dictionary.
    """
    from koopsim.utils.dictionary import (
        CompositeDictionary,
        IdentityDictionary,
        PolynomialDictionary,
        RBFDictionary,
    )

    dtype = config["type"]

    if dtype == "CompositeDictionary":
        # The CompositeDictionary serializes all its sub-dictionaries including
        # the auto-prepended identity at index 0. When we reconstruct, we need
        # to pass only the non-identity dictionaries to the constructor (since
        # CompositeDictionary auto-prepends its own identity).
        sub_configs = config["dictionaries"]
        dict_grp = f["dictionary"]

        # Reconstruct each sub-dictionary and restore its fitted state
        all_subs = []
        for i, sc in enumerate(sub_configs):
            sub_grp = dict_grp[f"sub_{i}"]
            sub_dict = _reconstruct_single_dictionary(sc, sub_grp)
            all_subs.append(sub_dict)

        # The first sub is always the auto-prepended IdentityDictionary.
        # CompositeDictionary's constructor filters out IdentityDictionary
        # instances and prepends its own, so we pass all sub-dicts and
        # let the constructor handle deduplication. But we need to restore
        # the identity's fitted state afterward.
        non_identity_subs = all_subs[1:]  # skip the serialized identity
        comp = CompositeDictionary(non_identity_subs)

        # Restore the identity dictionary fitted state
        identity_config = sub_configs[0]
        comp._identity._n_features = identity_config.get("n_features")

        # Restore fitted state on each non-identity sub-dictionary
        # (already done in _reconstruct_single_dictionary)

        comp._fitted = True
        return comp

    elif dtype == "IdentityDictionary":
        d = IdentityDictionary()
        d._n_features = config.get("n_features")
        return d

    elif dtype == "PolynomialDictionary":
        d = PolynomialDictionary(degree=config["degree"])
        # We need to restore the fitted state. Create a dummy dataset to fit.
        n_features = config["n_features"]
        dummy = np.zeros((1, n_features))
        d.fit(dummy)
        return d

    elif dtype == "RBFDictionary":
        d = RBFDictionary(
            n_centers=config["n_centers"],
            kernel=config["kernel"],
            gamma=config["gamma"],
        )
        # Restore fitted state from HDF5
        dict_grp = f["dictionary"]
        if "centers" in dict_grp:
            d._centers = np.array(dict_grp["centers"])
        if "gamma" in dict_grp.attrs:
            d._gamma = float(dict_grp.attrs["gamma"])
        else:
            d._gamma = config["gamma"]
        return d

    else:
        raise KoopSimError(f"Unknown dictionary type in file: {dtype}")


def _reconstruct_single_dictionary(config: dict, grp: h5py.Group):
    """Reconstruct a single non-composite dictionary from config and HDF5 group.

    Parameters
    ----------
    config : dict
        Configuration dictionary for this sub-dictionary.
    grp : h5py.Group
        HDF5 group containing this dictionary's data.

    Returns
    -------
    ObservableDictionary
        Reconstructed dictionary with fitted state restored.
    """
    from koopsim.utils.dictionary import (
        IdentityDictionary,
        PolynomialDictionary,
        RBFDictionary,
    )

    dtype = config["type"]

    if dtype == "IdentityDictionary":
        d = IdentityDictionary()
        d._n_features = config.get("n_features")
        return d

    elif dtype == "PolynomialDictionary":
        d = PolynomialDictionary(degree=config["degree"])
        n_features = config["n_features"]
        dummy = np.zeros((1, n_features))
        d.fit(dummy)
        return d

    elif dtype == "RBFDictionary":
        d = RBFDictionary(
            n_centers=config["n_centers"],
            kernel=config["kernel"],
            gamma=config["gamma"],
        )
        if "centers" in grp:
            d._centers = np.array(grp["centers"])
        if "gamma" in grp.attrs:
            d._gamma = float(grp.attrs["gamma"])
        else:
            d._gamma = config["gamma"]
        return d

    else:
        raise KoopSimError(f"Unknown dictionary type: {dtype}")
