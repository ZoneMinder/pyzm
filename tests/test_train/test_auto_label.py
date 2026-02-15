"""Tests for pyzm.train.auto_label."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.detection import BBox, Detection, DetectionResult
from pyzm.train.auto_label import _bbox_to_annotation, detections_to_annotations
from pyzm.train.dataset import Annotation


# ---------------------------------------------------------------------------
# _bbox_to_annotation
# ---------------------------------------------------------------------------

class TestBboxToAnnotation:
    def test_basic_conversion(self):
        ann = _bbox_to_annotation(100, 200, 300, 400, class_id=1, img_w=1000, img_h=1000)
        assert ann.class_id == 1
        assert ann.cx == pytest.approx(0.2)   # (100+300)/2 / 1000
        assert ann.cy == pytest.approx(0.3)   # (200+400)/2 / 1000
        assert ann.w == pytest.approx(0.2)    # (300-100) / 1000
        assert ann.h == pytest.approx(0.2)    # (400-200) / 1000

    def test_full_image_box(self):
        ann = _bbox_to_annotation(0, 0, 640, 480, class_id=0, img_w=640, img_h=480)
        assert ann.cx == pytest.approx(0.5)
        assert ann.cy == pytest.approx(0.5)
        assert ann.w == pytest.approx(1.0)
        assert ann.h == pytest.approx(1.0)

    def test_small_box(self):
        ann = _bbox_to_annotation(10, 10, 20, 20, class_id=0, img_w=640, img_h=480)
        assert ann.w == pytest.approx(10 / 640)
        assert ann.h == pytest.approx(10 / 480)


# ---------------------------------------------------------------------------
# detections_to_annotations
# ---------------------------------------------------------------------------

class TestDetectionsToAnnotations:
    def test_filters_by_target_classes(self):
        result = DetectionResult(detections=[
            Detection(label="person", confidence=0.9, bbox=BBox(10, 20, 110, 120)),
            Detection(label="car", confidence=0.8, bbox=BBox(200, 200, 400, 400)),
            Detection(label="cat", confidence=0.7, bbox=BBox(50, 50, 100, 100)),
        ])
        target = ["person", "car"]
        anns = detections_to_annotations(result, target, img_w=640, img_h=480)
        assert len(anns) == 2
        assert anns[0].class_id == 0  # person
        assert anns[1].class_id == 1  # car

    def test_case_insensitive_matching(self):
        result = DetectionResult(detections=[
            Detection(label="Person", confidence=0.9, bbox=BBox(10, 20, 110, 120)),
        ])
        anns = detections_to_annotations(result, ["person"], img_w=640, img_h=480)
        assert len(anns) == 1
        assert anns[0].class_id == 0

    def test_empty_detections(self):
        result = DetectionResult(detections=[])
        anns = detections_to_annotations(result, ["person"], img_w=640, img_h=480)
        assert anns == []

    def test_no_matching_classes(self):
        result = DetectionResult(detections=[
            Detection(label="dog", confidence=0.9, bbox=BBox(10, 20, 110, 120)),
        ])
        anns = detections_to_annotations(result, ["person", "car"], img_w=640, img_h=480)
        assert anns == []

    def test_correct_normalised_coords(self):
        result = DetectionResult(detections=[
            Detection(label="person", confidence=0.9, bbox=BBox(100, 120, 300, 360)),
        ])
        anns = detections_to_annotations(result, ["person"], img_w=640, img_h=480)
        ann = anns[0]
        # cx = (100+300)/2 / 640 = 200/640 = 0.3125
        assert ann.cx == pytest.approx(0.3125)
        # cy = (120+360)/2 / 480 = 240/480 = 0.5
        assert ann.cy == pytest.approx(0.5)
        # w = (300-100)/640 = 200/640 = 0.3125
        assert ann.w == pytest.approx(0.3125)
        # h = (360-120)/480 = 240/480 = 0.5
        assert ann.h == pytest.approx(0.5)
