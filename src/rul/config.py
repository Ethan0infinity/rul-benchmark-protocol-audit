from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of base and overrides."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using a dotted key such as data.subset."""
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def project_path(config: dict[str, Any], *parts: str) -> Path:
    """Resolve a path relative to the directory containing the config file."""
    cfg_path = Path(config.get("_config_path", ".")).resolve()
    explicit_root = config.get("_project_root") or config.get("project", {}).get("root_dir")
    if explicit_root:
        root = Path(explicit_root).resolve()
        return root.joinpath(*parts)

    root = None
    for candidate in [cfg_path.parent, *cfg_path.parents]:
        if (candidate / "src" / "rul").exists() and (candidate / "configs").exists():
            root = candidate
            break
    if root is None:
        root = cfg_path.parent.parent if cfg_path.parent.name == "configs" else cfg_path.parent
    return root.joinpath(*parts)
