"""Structured logging: JSON lines to logs/clipfarm.jsonl, readable lines to the console.

Secrets are redacted: any environment variable whose name contains KEY/TOKEN/SECRET/
PASSWORD/AUTH has its value replaced by *** in every message and extra field, and
strings shaped like common API keys are masked too.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH)", re.I)
_KEY_SHAPES = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)(x-mc-auth['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{12,}"),
]
_STD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}


def _secret_values() -> list[str]:
    vals = [v for k, v in os.environ.items() if _SECRET_NAME.search(k) and v and len(v) >= 6]
    return sorted(set(vals), key=len, reverse=True)


def redact(text: str, secrets: list[str] | None = None) -> str:
    if not isinstance(text, str):
        return text
    for s in secrets if secrets is not None else _secret_values():
        text = text.replace(s, "***")
    for pat in _KEY_SHAPES:
        if pat.groups:
            text = pat.sub(lambda m: m.group(1) + "***", text)
        else:
            text = pat.sub("***", text)
    return text


def _redact_obj(obj, secrets):
    if isinstance(obj, str):
        return redact(obj, secrets)
    if isinstance(obj, dict):
        return {k: ("***" if _SECRET_NAME.search(str(k)) and isinstance(v, str) and v else _redact_obj(v, secrets))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact_obj(v, secrets) for v in obj]
    return obj


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        secrets = _secret_values()
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - never let logging crash the pipeline
            msg = str(record.msg)
        record.msg = redact(msg, secrets)
        record.args = ()
        for key in list(vars(record)):
            if key not in _STD_ATTRS:
                setattr(record, key, _redact_obj(getattr(record, key), secrets))
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        extras = {k: v for k, v in vars(record).items() if k not in _STD_ATTRS and not k.startswith("_")}
        tail = " ".join(f"{k}={v}" for k, v in extras.items() if k in ("job", "stage", "fmt", "brand", "provider"))
        line = f"{ts} {record.levelname[0]} {record.getMessage()}"
        if tail:
            line += f"  [{tail}]"
        if record.exc_info:
            line += "\n" + redact(self.formatException(record.exc_info))
        return line


_configured = False


def setup_logging(log_dir: Path | str | None = None, verbose: bool = False, quiet: bool = False) -> None:
    global _configured
    root = logging.getLogger("clipfarm")
    if _configured:
        return
    root.setLevel(logging.DEBUG)
    root.propagate = False
    redactor = RedactingFilter()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO))
    console.setFormatter(ConsoleFormatter())
    console.addFilter(redactor)
    root.addHandler(console)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(log_dir / "clipfarm.jsonl", maxBytes=10_000_000,
                                                  backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonFormatter())
        fh.addFilter(redactor)
        root.addHandler(fh)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("clipfarm") else f"clipfarm.{name}")
