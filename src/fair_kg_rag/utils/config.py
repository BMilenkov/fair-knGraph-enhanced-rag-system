"""Configuration loading using OmegaConf with .env support."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load .env file on import — looks in cwd and project root
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, merging with base config and CLI overrides."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    if "base" in cfg:
        base_path = config_path.parent / cfg.base
        if base_path.exists():
            cfg = OmegaConf.merge(OmegaConf.load(base_path), cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def parse_cli_overrides() -> list[str]:
    """Extract key=value overrides from sys.argv."""
    return [arg for arg in sys.argv[1:] if "=" in arg and not arg.startswith("-")]
