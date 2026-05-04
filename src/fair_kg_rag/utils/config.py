"""Configuration loading utilities using OmegaConf."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file with optional CLI overrides.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: List of key=value strings to override config values.

    Returns:
        Merged DictConfig with base and override values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # If config references a base config, merge with it
    if "base" in cfg:
        base_path = config_path.parent / cfg.base
        if base_path.exists():
            base_cfg = OmegaConf.load(base_path)
            cfg = OmegaConf.merge(base_cfg, cfg)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def load_experiment_config(
    experiment_path: str | Path,
    base_dir: str | Path | None = None,
) -> DictConfig:
    """Load an experiment config, merging with all referenced base configs.

    Args:
        experiment_path: Path to the experiment YAML file.
        base_dir: Base directory for resolving relative config paths.

    Returns:
        Fully merged DictConfig.
    """
    experiment_path = Path(experiment_path)
    if base_dir is None:
        base_dir = experiment_path.parent.parent  # configs/ directory

    base_dir = Path(base_dir)
    merged = OmegaConf.create({})

    # Load base.yaml if it exists
    base_yaml = base_dir / "base.yaml"
    if base_yaml.exists():
        merged = OmegaConf.merge(merged, OmegaConf.load(base_yaml))

    # Load the experiment config on top
    exp_cfg = OmegaConf.load(experiment_path)
    merged = OmegaConf.merge(merged, exp_cfg)

    return merged


def parse_cli_overrides() -> list[str]:
    """Extract key=value overrides from sys.argv.

    Returns:
        List of override strings like ["key=value", ...].
    """
    overrides = []
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
    return overrides


def get_config_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Safely get a nested config value using dot notation.

    Args:
        cfg: The configuration object.
        key: Dot-separated key path (e.g., "retrieval.top_k").
        default: Default value if key not found.

    Returns:
        The config value or default.
    """
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default
