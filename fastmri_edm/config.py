"""YAML configuration with dotted command-line overrides."""
from pathlib import Path
from types import SimpleNamespace

import yaml


def to_namespace(d):
    """Recursively convert dicts to namespaces."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    return d


def to_dict(ns):
    """Recursively convert namespaces back to dicts."""
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return ns


def parse_value(text: str):
    """YAML scalar, with plain numbers like 1e-4 read as floats."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_override(cfg: dict, item: str) -> None:
    """Set a nested key from 'a.b=value'."""
    key, text = item.split("=", 1)
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = parse_value(text)


def load_config(path, overrides=()) -> SimpleNamespace:
    """Load a YAML config and apply overrides."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        apply_override(cfg, item)
    return to_namespace(cfg)


def save_config(cfg: SimpleNamespace, path) -> None:
    """Write a config back to YAML."""
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
