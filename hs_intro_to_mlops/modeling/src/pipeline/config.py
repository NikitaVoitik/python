from omegaconf import OmegaConf, DictConfig
from pathlib import Path


def load_config(path: str = "config.yaml") -> DictConfig:
    """Load config.yaml and merge any CLI overrides."""
    cfg = OmegaConf.load(path)
    cli_cfg = OmegaConf.from_cli()
    return OmegaConf.merge(cfg, cli_cfg)
