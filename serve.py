#!/usr/bin/env python3
"""Lightweight HTTP server for the docs directory."""

from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the docs directory so index.html can be viewed in a browser.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind the HTTP server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server (default: 8000).",
    )
    parser.add_argument(
        "--directory",
        default=Path(__file__).parent / "docs",
        type=Path,
        help="Directory to serve (default: ./docs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.exists():
        raise SystemExit(f"Directory not found: {directory}")

    os.chdir(directory)

    server_address = (args.host, args.port)
    handler_class = SimpleHTTPRequestHandler

    with ThreadingHTTPServer(server_address, handler_class) as httpd:
        print(
            f"Serving {directory} at http://{args.host}:{args.port}/ (Ctrl+C to stop)",
            flush=True,
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
