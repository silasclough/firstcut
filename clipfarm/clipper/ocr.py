"""Read the words burned into a picture (a creator's own "SUBSCRIBE" bar, an @handle, a channel logo).

macOS has a text reader built in (Vision); a 20-line Swift helper (`cf_ocr.swift`) is compiled once into data/models/cf_ocr and reused. Where Swift is not
installed, `tesseract` is used if there is one. With neither, `read()` returns None ("could not look"), and the checks that depend on it are skipped, never failed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.ocr")
SRC = Path(__file__).with_name("cf_ocr.swift")
BIN = Path(__file__).resolve().parents[2] / "data" / "models" / "cf_ocr"


def helper() -> Path | None:
    """The compiled Vision helper: the app ships it ready-made (built by scripts/package_videobuilder.sh into its bin folder) and never
    compiles anything; elsewhere it is built on first use (about a minute) when the Command Line Tools are installed. None = no text reader."""
    import os
    shipped = os.environ.get("VIDEOBUILDER_BIN")
    if shipped:                                          # the app: its own copy, or none (never Apple's "install developer tools?" box)
        b = Path(shipped) / "cf_ocr"
        return b if b.is_file() and os.access(b, os.X_OK) else None
    if BIN.exists() and BIN.stat().st_mtime >= SRC.stat().st_mtime:
        return BIN
    swiftc = shutil.which("swiftc")
    dev = subprocess.run(["/usr/bin/xcode-select", "-p"], capture_output=True).returncode == 0 if Path("/usr/bin/xcode-select").exists() else False
    if not swiftc or not dev:                            # (/usr/bin/swiftc without the tools is only Apple's installer stub)
        return None
    try:
        BIN.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([swiftc, "-O", str(SRC), "-o", str(BIN)], capture_output=True, timeout=900, check=True)
        return BIN
    except Exception as e:  # noqa: BLE001 - no helper just means no OCR
        log.warning(f"could not build the text reader: {e}")
        return None


def _tesseract(images: list[Path]) -> list[list[dict]] | None:
    t = ffmpeg.which("tesseract")
    if not t:
        return None
    from PIL import Image
    out = []
    for im in images:
        try:
            iw, ih = Image.open(im).size
            r = subprocess.run([t, str(im), "-", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=60).stdout.splitlines()[1:]
        except Exception:  # noqa: BLE001
            out.append([])
            continue
        rows = []
        for ln in r:
            p = ln.split("\t")
            if len(p) >= 12 and p[11].strip() and float(p[10]) > 40:
                rows.append({"t": p[11].strip(), "x": int(p[6]) / iw, "y": int(p[7]) / ih, "w": int(p[8]) / iw, "h": int(p[9]) / ih, "c": float(p[10]) / 100})
        out.append(rows)
    return out


def read(images: list[Path]) -> list[list[dict]] | None:
    """-> for each image the text lines found ({"t": text, "x","y","w","h": position as a share of the picture, "c": confidence}); None if nothing can read text here."""
    if not images:
        return []
    h = helper()
    if h:
        try:
            p = subprocess.run([str(h), *map(str, images)], capture_output=True, text=True, timeout=120)
            lines = [json.loads(x) for x in p.stdout.splitlines() if x.strip().startswith("[")]
            if len(lines) == len(images):
                return lines
        except Exception as e:  # noqa: BLE001
            log.debug(f"text reader failed: {e}")
    return _tesseract(images)
