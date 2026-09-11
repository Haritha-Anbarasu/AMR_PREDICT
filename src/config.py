"""
src/config.py

Loads config.yaml once and exposes a simple dot-path getter so every
other module reads parameters the same way instead of hardcoding values
or re-parsing YAML repeatedly.

Usage:
    from src.config import cfg

    identity = cfg("dataset.cdhit_identity")
    seed = cfg("random_seed")
    identity_with_fallback = cfg("dataset.cdhit_identity", default=0.90)
"""
from pathlib import Path
from functools import lru_cache
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def _load_raw_config(config_path: str = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. This file holds every tunable "
            "pipeline parameter (CD-HIT identity, split ratios, random seed, "
            "k-mer size, etc.) — copy the one shipped with the project scaffold "
            "into the project root, or restore it from version control."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def cfg(key_path: str, default=None):
    """
    Dot-path getter into config.yaml, e.g. cfg("dataset.cdhit_identity").
    Returns `default` if the key path doesn't exist, rather than raising,
    so optional/new config sections degrade gracefully in older configs.
    """
    node = _load_raw_config()
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def get_path(key: str) -> Path:
    """Convenience wrapper for paths.* config entries, returned as Path objects."""
    value = cfg(f"paths.{key}")
    if value is None:
        raise KeyError(f"No 'paths.{key}' entry in config.yaml")
    return Path(value)


def reload_config():
    """Clear the cached config — mainly useful in tests or interactive notebooks
    after editing config.yaml without restarting the kernel."""
    _load_raw_config.cache_clear()
