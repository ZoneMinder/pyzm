"""Shared fixtures and helpers for e2e detection tests.

These tests use a real bird.jpg image and real YOLO models on disk.
They require:
  - OpenCV (cv2), numpy, shapely
  - Models at /var/lib/zmeventnotification/models/ (yolov4 at minimum)

Run all e2e tests:
    python -m pytest tests/test_e2e/ -v --tb=short
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIRD_IMAGE = str(Path(__file__).parent / "bird.jpg")
BASE_PATH = "/var/lib/zmeventnotification/models"


# ---------------------------------------------------------------------------
# Module-level skip conditions
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Skip e2e tests if models or image are missing."""
    skip_models = pytest.mark.skip(reason=f"Model base path {BASE_PATH} not found")
    skip_image = pytest.mark.skip(reason=f"Test image {BIRD_IMAGE} not found")
    for item in items:
        if "test_e2e" in str(item.fspath):
            if not os.path.isdir(BASE_PATH):
                item.add_marker(skip_models)
            if not os.path.isfile(BIRD_IMAGE):
                item.add_marker(skip_image)


# ---------------------------------------------------------------------------
# Helpers (importable by test files)
# ---------------------------------------------------------------------------

def load_image(path: str = BIRD_IMAGE):
    import cv2
    img = cv2.imread(path)
    assert img is not None, f"Failed to load {path}"
    return img


def det(label, x1, y1, x2, y2, conf=0.9, model_name="test"):
    from pyzm.models.detection import BBox, Detection
    return Detection(
        label=label,
        confidence=conf,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        model_name=model_name,
    )


def find_one_model():
    """Find the first available model in BASE_PATH."""
    from pyzm.ml.detector import _discover_models
    from pyzm.models.config import Processor
    models = _discover_models(Path(BASE_PATH), Processor.CPU)
    assert len(models) > 0, "No models found in base path"
    return models[0].name


def resolve(model_name):
    from pyzm.ml.detector import _resolve_model_name
    from pyzm.models.config import Processor
    return _resolve_model_name(model_name, Path(BASE_PATH), Processor.CPU)


# ---------------------------------------------------------------------------
# Serve helpers
# ---------------------------------------------------------------------------

def start_serve(models, port, extra_args=None):
    cmd = [
        sys.executable, "-m", "pyzm.serve",
        "--models", *models,
        "--base-path", BASE_PATH,
        "--processor", "cpu",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="/home/arjunrc/fiddle/pyzm",
        env={**os.environ, "PYTHONPATH": "/home/arjunrc/fiddle/pyzm"},
    )


def wait_for_serve(port, timeout=90):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def stop_serve(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
