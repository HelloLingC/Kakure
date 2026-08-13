"""CLI entry point — launch the Kakure web UI."""

from __future__ import annotations

import logging


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="kakure",
        description="Kakure - ASMR Japanese-to-Chinese bilingual voice overlay tool",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server hostname (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7530,
        help="Server port (default: 7530)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open a browser window on startup",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (for development)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.no_browser:
        import threading
        import webbrowser

        browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        url = f"http://{browser_host}:{args.port}"
        logging.getLogger(__name__).info("Opening Kakure in your browser: %s", url)
        threading.Timer(1.25, webbrowser.open, args=(url,)).start()

    # Route all model downloads into the project folder when model_dir is set
    # (portable / integrated-package mode). Must run before any model library
    # is imported, hence before uvicorn starts.
    from kakure.config import apply_model_env, load_settings

    apply_model_env(load_settings())

    import uvicorn

    uvicorn.run(
        "kakure.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
