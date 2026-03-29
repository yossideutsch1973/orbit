"""Experiment entrypoint: python -m experiments.run --config <name>"""

import argparse
import sys
from pathlib import Path

import torch

from baselines.mlp import MLPActorCritic
from dkn.network import KoopmanNet
from experiments.config import ExperimentConfig
from training.runner import TrainResult, count_parameters, train_ppo


def find_config(name: str) -> Path:
    """Find config YAML by name or path."""
    p = Path(name)
    if p.exists():
        return p
    configs_dir = Path(__file__).parent / "configs"
    p = configs_dir / f"{name}.yaml"
    if p.exists():
        return p
    root = Path(__file__).parent.parent.parent
    p = root / f"{name}.yaml"
    if p.exists():
        return p
    print(f"Config not found: {name}")
    sys.exit(1)


def print_results(name: str, results: dict[str, TrainResult]) -> None:
    """Print comparison table."""
    print(f"\nEXPERIMENT: {name}")
    header = f"{'method':<10} | {'params':>8} | {'reward_mean':>11} | {'reward_std':>10} | {'best':>6} | {'wall_time':>9}"
    print(header)
    print("-" * len(header))
    for method, r in results.items():
        print(
            f"{method:<10} | {r.total_params:>8} | {r.final_reward_mean:>11.1f} "
            f"| {r.final_reward_std:>10.1f} | {r.best_reward:>6.1f} | {r.wall_time:>8.1f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DKN experiment")
    parser.add_argument("--config", required=True, help="Config name or path")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--arch", default="both", choices=["dkn", "mlp", "both"])
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")

    cfg = ExperimentConfig.from_yaml(find_config(args.config))
    print(f"Experiment: {cfg.name} | Env: {cfg.env}")

    networks: dict[str, torch.nn.Module] = {}
    if args.arch in ("dkn", "both"):
        networks["DKN"] = KoopmanNet(cfg.build_dkn_config(), device)
    if args.arch in ("mlp", "both"):
        networks["MLP"] = MLPActorCritic(cfg.state_dim, cfg.action_dim, cfg.mlp_hidden, device)

    for name, net in networks.items():
        print(f"  {name}: {count_parameters(net)} params")

    ppo_cfg = cfg.build_ppo_config()
    results: dict[str, TrainResult] = {}
    for arch_name, net in networks.items():
        print(f"\n{'=' * 50}\nTraining {arch_name}\n{'=' * 50}")
        results[arch_name] = train_ppo(
            network=net,
            env_name=cfg.env,
            total_steps=cfg.training.total_steps,
            steps_per_update=cfg.training.steps_per_update,
            ppo_cfg=ppo_cfg,
            gamma=cfg.training.gamma,
            gae_lambda=cfg.training.gae_lambda,
            device=device,
            seed=cfg.seed,
            log_interval=cfg.logging.log_interval,
            is_dkn=(arch_name == "DKN"),
        )

    print_results(cfg.name, results)


if __name__ == "__main__":
    main()
