"""Experiment configuration dataclasses and YAML parser."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from dkn.network import DKNConfig
from training.ppo import PPOConfig


@dataclass
class DKNArchConfig:
    """DKN architecture parameters (from YAML dkn: section)."""

    n_layers: int
    neurons_per_layer: int
    d_lift: int
    d_out_per_neuron: int
    head_hidden: int = 0
    broadcast: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    algorithm: str
    total_steps: int
    steps_per_update: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    ppo_epochs: int
    minibatch_size: int
    max_grad_norm: float
    entropy_coef: float
    value_coef: float
    spectral_coef: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_interval: int
    spectra_interval: int
    save_model: bool


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    env: str
    state_dim: int
    action_dim: int
    seed: int
    dkn: DKNArchConfig
    mlp_hidden: int
    training: TrainingConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            env=raw["env"],
            state_dim=raw["state_dim"],
            action_dim=raw["action_dim"],
            seed=raw["seed"],
            dkn=DKNArchConfig(**raw["dkn"]),
            mlp_hidden=raw["mlp_baseline"]["hidden"],
            training=TrainingConfig(**raw["training"]),
            logging=LoggingConfig(**raw["logging"]),
        )

    def build_dkn_config(self) -> DKNConfig:
        """Build DKNConfig for KoopmanNet construction."""
        return DKNConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            **dataclasses.asdict(self.dkn),
        )

    def build_ppo_config(self) -> PPOConfig:
        """Build PPOConfig for PPO updater."""
        t = self.training
        return PPOConfig(
            lr=t.lr,
            clip_eps=t.clip_eps,
            ppo_epochs=t.ppo_epochs,
            minibatch_size=t.minibatch_size,
            max_grad_norm=t.max_grad_norm,
            entropy_coef=t.entropy_coef,
            value_coef=t.value_coef,
            spectral_coef=t.spectral_coef,
        )
