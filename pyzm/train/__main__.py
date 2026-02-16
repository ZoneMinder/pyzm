"""CLI entry point: ``python -m pyzm.train``."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pyzm YOLO Fine-Tuning Training UI",
        prog="python -m pyzm.train",
    )
    ap.add_argument(
        "--base-path",
        default="/var/lib/zmeventnotification/models",
        help="Model base path for auto-labeling (default: /var/lib/zmeventnotification/models)",
    )
    ap.add_argument(
        "--workspace-dir",
        default=None,
        help="Training workspace (default: ~/.pyzm/training)",
    )
    ap.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port (default: 8501)",
    )
    ap.add_argument(
        "--processor",
        default="gpu",
        choices=["cpu", "gpu"],
        help="Processor for auto-labeling (default: gpu)",
    )
    args = ap.parse_args()

    from pyzm.train import check_dependencies
    check_dependencies()

    import subprocess

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        _app_path(),
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--theme.base", "dark",
        "--theme.primaryColor", "#a855f7",
        "--theme.backgroundColor", "#0d1117",
        "--theme.secondaryBackgroundColor", "#161b22",
        "--theme.textColor", "#f0f6fc",
        "--theme.font", "sans serif",
        "--",
        "--base-path", args.base_path,
        "--processor", args.processor,
    ]
    if args.workspace_dir:
        cmd.extend(["--workspace-dir", args.workspace_dir])

    sys.exit(subprocess.call(cmd))


def _app_path() -> str:
    """Return the absolute path to the Streamlit app module."""
    from pathlib import Path

    return str(Path(__file__).with_name("app.py"))


if __name__ == "__main__":
    main()
