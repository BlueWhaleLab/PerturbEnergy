"""
Config loading: one YAML file per dataset on top of `configs/default.yaml`, plus dotted
command-line overrides so a sweep never needs a new file.

    cfg = load("configs/replogle.yaml", ["model.z_dim=256", "loss.beta_kl_a=1e-4"])
    cfg.model.z_dim            # attribute access, nested
    cfg["model"]["z_dim"]      # or dict access
    cfg.to_dict()              # plain dict, JSON-serialisable (what a run records)

A dataset config only lists what it changes; everything else falls through to the
default. Overrides are parsed as YAML, so `true`, `1e-4`, `[1,2]` and `null` all work.
"""
import copy
import os

import yaml

from .paths import PROJECT_ROOT

DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


class Cfg(dict):
    """A dict whose nested dicts are also Cfg, with attribute access."""

    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = Cfg(v) if isinstance(v, dict) else v

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Cfg) else v) for k, v in self.items()}


def _deep_update(base, other):
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_dotted(d, dotted, value):
    """`loss.beta_kl_a=1e-4` -> d["loss"]["beta_kl_a"] = 1e-4. The key must already
    exist: a typo in a sweep would otherwise silently do nothing."""
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"--set {dotted}: '{k}' is not a config section")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"--set {dotted}: unknown key '{keys[-1]}' "
                       f"(known: {sorted(node)})")
    node[keys[-1]] = value


def load(path=None, overrides=(), defaults=DEFAULT_CONFIG):
    """default.yaml <- `path` <- `overrides` (each 'a.b.c=yaml_value')."""
    cfg = yaml.safe_load(open(defaults)) if defaults and os.path.exists(defaults) else {}
    if path and os.path.abspath(path) != os.path.abspath(defaults or ""):
        _deep_update(cfg, yaml.safe_load(open(path)) or {})
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    return Cfg(cfg)


def dataset_config(dataset, overrides=()):
    """The config for a named dataset (`configs/<dataset>.yaml` if it exists)."""
    p = os.path.join(PROJECT_ROOT, "configs", f"{dataset}.yaml")
    return load(p if os.path.exists(p) else None, overrides)


def add_args(ap):
    """Standard config flags for every script."""
    ap.add_argument("--config", default=None, help="YAML on top of configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=x",
                    help="dotted config overrides, e.g. --set model.z_dim=256")
    return ap


def save(cfg, path):
    import json
    with open(path, "w") as f:
        json.dump(cfg.to_dict() if isinstance(cfg, Cfg) else cfg, f, indent=2, default=str)
    return path


def clone(cfg):
    return Cfg(copy.deepcopy(cfg.to_dict() if isinstance(cfg, Cfg) else cfg))
