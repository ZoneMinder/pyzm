"""E2E tests for object detection on live ZM events.

These tests require BOTH a live ZM server AND ML models on disk.
Skipped when either prerequisite is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.zm_e2e

_BASE_PATH = "/var/lib/zmeventnotification/models"
_models_available = os.path.isdir(_BASE_PATH)

try:
    import cv2  # noqa: F401

    _cv2_available = True
except ImportError:
    _cv2_available = False

skip_no_models = pytest.mark.skipif(
    not _models_available, reason=f"Model path {_BASE_PATH} not found"
)
skip_no_cv2 = pytest.mark.skipif(not _cv2_available, reason="cv2 not installed")


def _make_detector():
    """Build a Detector using auto-discovered models."""
    from pyzm.ml.detector import Detector

    return Detector(base_path=_BASE_PATH)


@skip_no_models
@skip_no_cv2
class TestDetectEvent:
    """detect_event() against a live ZM server with real models."""

    def test_pipeline_runs_on_any_event(self, zm_client, any_event, e2e_summary):
        """detect_event() should return a DetectionResult without crashing."""
        from pyzm.models.detection import DetectionResult

        det = _make_detector()
        result = det.detect_event(zm_client, any_event.id)
        assert isinstance(result, DetectionResult)
        labels = [f"{d.label} ({d.confidence:.0%})" for d in result.detections]
        e2e_summary.setdefault("Detection (any event)", []).append(
            ("Event " + str(any_event.id), ", ".join(labels) if labels else "(none)"))

    def test_detections_on_object_event(self, zm_client, object_event, e2e_summary):
        """An event with 'detected' in notes should yield actual detections."""
        from pyzm.models.detection import DetectionResult

        det = _make_detector()
        result = det.detect_event(zm_client, object_event.id)
        assert isinstance(result, DetectionResult)
        assert len(result.detections) > 0, (
            f"Expected detections on object event {object_event.id} "
            f"but got none"
        )
        labels = [f"{d.label} ({d.confidence:.0%})" for d in result.detections]
        e2e_summary.setdefault("Detection (object event)", []).append(
            ("Event " + str(object_event.id), ", ".join(labels)))

    def test_detection_fields(self, zm_client, object_event):
        """Each detection should have label, confidence, and bbox."""
        det = _make_detector()
        result = det.detect_event(zm_client, object_event.id)
        if not result.detections:
            pytest.skip("No detections returned")
        d = result.detections[0]
        assert isinstance(d.label, str) and len(d.label) > 0
        assert isinstance(d.confidence, float) and 0.0 < d.confidence <= 1.0
        assert d.bbox is not None
        assert d.bbox.x2 > d.bbox.x1
        assert d.bbox.y2 > d.bbox.y1
