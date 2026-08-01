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
        default="0.0.0.0",
        help="Server hostname (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
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

    import uvicorn

    uvicorn.run(
        "kakure.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
