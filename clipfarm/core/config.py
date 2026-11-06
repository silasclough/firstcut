"""Config loading: config/config.yaml deep-merged with config/config.local.yaml, plus .env."""

from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(os.environ.get("CLIPFARM_ROOT") or Path(__file__).resolve().parents[2])


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_env(root: Path = ROOT) -> None:
    """Load .env without overriding variables already set in the environment."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:  # minimal parser so doctor.py still works without deps
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Config:
    """Thin wrapper around the merged config dict."""

    def __init__(self, data: dict, root: Path = ROOT, source: Path | None = None):
        self.data = data
        self.root = root
        self.source = source

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        raw = self.get(f"paths.{key}")
        if raw is None:
            raise KeyError(f"paths.{key} is not configured")
        p = Path(os.path.expanduser(str(raw)))
        return p if p.is_absolute() else self.root / p

    def resolve(self, raw: str | Path) -> Path:
        p = Path(os.path.expanduser(str(raw)))
        return p if p.is_absolute() else self.root / p

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.get("timezone", "UTC"))

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def fmt(self, name: str) -> dict:
        formats = self.get("formats", {}) or {}
        if name not in formats:
            raise KeyError(f"unknown format '{name}' (known: {', '.join(formats)})")
        return formats[name]

    def brand(self, name: str) -> dict:
        brands = self.get("brands", {}) or {}
        if name not in brands:
            raise KeyError(f"unknown brand '{name}' (known: {', '.join(brands)})")
        return brands[name]

    def enabled_formats(self) -> list[str]:
        return [n for n, f in (self.get("formats", {}) or {}).items() if f.get("enabled")]

    def brand_for_format(self, fmt_name: str) -> str:
        return self.fmt(fmt_name)["brand"]

    def ensure_dirs(self) -> None:
        for key in ("queue", "state", "work", "review", "ready", "posted", "rejected", "public",
                    "logs", "reports", "minecraft_footage", "music", "placeholder", "streamer_inbox", "story_inbox"):
            try:
                self.path(key).mkdir(parents=True, exist_ok=True)
            except KeyError:
                pass


def machine_ram_gb() -> float:
    try:
        import subprocess
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                  timeout=5).stdout.strip()) / 2**30
    except Exception:  # noqa: BLE001
        return 0.0


def pick_profile(profiles: list[dict], ram_gb: float, forced: str | None = None) -> dict | None:
    """The profile named `forced`, else the one with the largest min_ram_gb <= installed RAM."""
    if forced:
        return next((p for p in profiles if p.get("name") == forced), None)
    fitting = [p for p in profiles if float(p.get("min_ram_gb", 0)) <= ram_gb]
    return max(fitting, key=lambda p: float(p.get("min_ram_gb", 0)), default=None)


def load_config(path: str | Path | None = None, overrides: dict | None = None,
                root: Path = ROOT) -> Config:
    load_env(root)
    cfg_path = Path(path or os.environ.get("CLIPFARM_CONFIG") or root / "config" / "config.yaml")
    data = yaml.safe_load(cfg_path.read_text()) or {}
    # machine profile (by installed RAM, or CLIPFARM_PROFILE) -> then config.local.yaml wins
    prof = pick_profile(data.get("machine_profiles") or [], machine_ram_gb(), os.environ.get("CLIPFARM_PROFILE"))
    if prof:
        data = deep_merge(data, prof.get("overrides") or {})
        data["active_profile"] = prof.get("name")
    local = cfg_path.with_name("config.local.yaml")
    if local.exists():
        data = deep_merge(data, yaml.safe_load(local.read_text()) or {})
    overlay = os.environ.get("CLIPFARM_OVERLAY")                 # a second app on the same machine (config/work.yaml: the work Video Builder)
    if overlay:
        op = Path(overlay) if Path(overlay).is_absolute() else root / overlay
        data = deep_merge(data, yaml.safe_load(op.read_text()) or {})
    if overrides:
        data = deep_merge(data, overrides)
    return Config(data, root=root, source=cfg_path)
