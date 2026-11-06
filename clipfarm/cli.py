"""FirstCut's engine commands:  python -m clipfarm studio | mcp | catalog

The app (app/VideoBuilder) starts `studio`; AI apps start `mcp` through the firstcut-mcp launcher; `catalog` runs the footage
catalogue without the app. (The engine package is called clipfarm for history.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def cmd_mcp(args) -> int:
    """MCP over stdio: only protocol on stdout, so logging goes to stderr (clipper/mcp_server.py)."""
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    from .clipper import mcp_server
    return mcp_server.main()


def cmd_studio(args) -> int:
    """FirstCut (the app) starts its engine with this: its settings become overlay.yaml, the models go where the person chose,
    our own Ollama starts when the Mac has none, then the app page and the engine are served on --port."""
    import signal

    from .clipper import studio
    from .core.config import load_config
    from .core.log import setup_logging
    root = studio.home()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(studio.environment(root))
    studio.keys_into_env()                                               # stock-footage keys from the Keychain (Settings › Online sources)
    if (root / "runtime" / "ollama" / "ollama").exists():
        try:
            studio.OllamaRunner(root).ensure()
        except Exception as e:  # noqa: BLE001 - Setup shows it; the rest of the app still opens
            print(f"Ollama didn't start: {e}", file=sys.stderr)
    os.environ["CLIPFARM_OVERLAY"] = str(studio.write_overlay(root))

    def stop(*_):
        studio.OllamaRunner.stop()
        os._exit(0)
    signal.signal(signal.SIGTERM, stop)
    (root / "server.json").write_text(json.dumps({"port": args.port, "pid": os.getpid(), "started": int(__import__("time").time())}))   # (vb finds it here)
    cfg = load_config()
    setup_logging(cfg.path("logs"))
    from .clipper.server import serve
    try:
        serve(cfg, host="127.0.0.1", port=args.port)
    finally:
        studio.OllamaRunner.stop()
    return 0


def cmd_catalog(args) -> int:
    """The footage catalog (clipper/catalog.py) without the server: for testing and one-off runs. The work Video Builder runs it itself
    (CLIPFARM_OVERLAY=config/work.yaml selects the work drive); don't run this while that server is cataloguing."""
    import time as _t

    from .clipper.catalog import Catalog
    from .core.config import load_config
    from .core.log import setup_logging
    cfg = load_config()
    setup_logging(cfg.path("logs"))
    cat = Catalog(cfg)
    if not cat.enabled():
        print("no catalog on this machine (catalog.roots is empty; the work drive is in config/work.yaml)", file=sys.stderr)
        return 2
    deadline = _t.time() + args.hours * 3600 if args.hours else None
    prog = (lambda k, n, what: print(f"[{k + 1}/{n}] {what}", flush=True))
    if args.action == "index":
        print(cat.index(deadline, limit=args.limit, progress=prog))
    elif args.action == "describe":
        print(cat.describe(deadline, limit=args.limit, progress=prog))
    elif args.action == "search":
        for sh in cat.search(" ".join(args.text or []), n=args.limit or 12):
            print(f"{sh['score']:.3f} p={sh['chance']:.3f} {sh['start']:7.1f}-{sh['end']:<7.1f} {sh['rel']}  {sh.get('what') or ''}")
    elif args.action == "scan":
        found = cat.scan()
        print(f"{len(found)} files ({sum(1 for f in found if f['kind'] == 'video')} videos), {len(cat.todo(found))} not catalogued yet")
    else:
        print(json.dumps(cat.stats(), indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="firstcut-engine", description="FirstCut's engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("studio", help="FirstCut (the app) starts its engine with this (app/VideoBuilder)")
    p.add_argument("--port", type=int, default=8798)
    p.set_defaults(fn=cmd_studio)

    p = sub.add_parser("mcp", help="FirstCut for AI apps: the MCP server on stdin/stdout (the app's launcher firstcut-mcp runs this)")
    p.set_defaults(fn=cmd_mcp)

    p = sub.add_parser("catalog", help="the footage catalog: status | scan | index | describe | search <words> (CLIPFARM_OVERLAY=config/work.yaml for the work drive)")
    p.add_argument("action", choices=["status", "scan", "index", "describe", "search"])
    p.add_argument("text", nargs="*", help="search: what the shot shows")
    p.add_argument("--hours", type=float, help="index / describe: stop after this long")
    p.add_argument("--limit", type=int, help="index / describe: at most this many files; search: how many results")
    p.set_defaults(fn=cmd_catalog)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
