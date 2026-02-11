"""Tests for pyzm.models.detection -- detection result models."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from pyzm.models.detection import BBox, Detection, DetectionResult


# ===================================================================
# TestBBox
# ===================================================================

class TestBBox:
    """Tests for the BBox frozen dataclass."""

    def test_creation(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.x1 == 10
        assert bb.y1 == 20
        assert bb.x2 == 50
        assert bb.y2 == 80

    def test_width(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.width == 40

    def test_height(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.height == 60

    def test_area(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.area == 40 * 60  # 2400

    def test_center(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.center == (30, 50)

    def test_center_odd_values(self):
        bb = BBox(x1=0, y1=0, x2=99, y2=99)
        assert bb.center == (49, 49)

    def test_as_list(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        assert bb.as_list() == [10, 20, 50, 80]

    def test_as_polygon_coords(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        coords = bb.as_polygon_coords()
        assert coords == [
            (10, 20),  # top-left
            (50, 20),  # top-right
            (50, 80),  # bottom-right
            (10, 80),  # bottom-left
        ]

    def test_frozen_immutability(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        with pytest.raises(AttributeError):
            bb.x1 = 100

    def test_frozen_immutability_y2(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        with pytest.raises(AttributeError):
            bb.y2 = 200

    def test_zero_area_bbox(self):
        bb = BBox(x1=10, y1=10, x2=10, y2=10)
        assert bb.area == 0
        assert bb.width == 0
        assert bb.height == 0


# ===================================================================
# TestDetection
# ===================================================================

class TestDetection:
    """Tests for the Detection frozen dataclass."""

    def test_creation(self):
        bb = BBox(x1=10, y1=20, x2=50, y2=80)
        det = Detection(label="person", confidence=0.95, bbox=bb)
        assert det.label == "person"
        assert det.confidence == 0.95
        assert det.bbox == bb
        assert det.model_name == ""
        assert det.detection_type == "object"

    def test_creation_with_all_fields(self):
        bb = BBox(x1=0, y1=0, x2=100, y2=100)
        det = Detection(
            label="face_john",
            confidence=0.88,
            bbox=bb,
            model_name="dlib",
            detection_type="face",
        )
        assert det.model_name == "dlib"
        assert det.detection_type == "face"

    def test_matches_pattern_simple(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        assert det.matches_pattern("person") is True
        assert det.matches_pattern("car") is False

    def test_matches_pattern_regex(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        assert det.matches_pattern("(person|car)") is True
        assert det.matches_pattern("per.*") is True
        assert det.matches_pattern("^per") is True
        assert det.matches_pattern("^car") is False

    def test_matches_pattern_wildcard(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        assert det.matches_pattern(".*") is True

    def test_frozen_immutability(self):
        bb = BBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.9, bbox=bb)
        with pytest.raises(AttributeError):
            det.label = "car"


# ===================================================================
# TestDetectionResult
# ===================================================================

class TestDetectionResult:
    """Tests for the DetectionResult aggregate dataclass."""

    def _make_detections(self):
        return [
            Detection(
                label="person", confidence=0.97,
                bbox=BBox(x1=10, y1=20, x2=50, y2=80),
                model_name="yolov4",
            ),
            Detection(
                label="car", confidence=0.85,
                bbox=BBox(x1=60, y1=30, x2=90, y2=70),
                model_name="yolov4",
            ),
        ]

    def test_matched_true(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.matched is True

    def test_matched_false_empty(self):
        dr = DetectionResult()
        assert dr.matched is False

    def test_labels_property(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.labels == ["person", "car"]

    def test_boxes_property(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.boxes == [[10, 20, 50, 80], [60, 30, 90, 70]]

    def test_confidences_property(self):
        dr = DetectionResult(detections=self._make_detections())
        assert dr.confidences == [0.97, 0.85]

    def test_summary(self):
        dr = DetectionResult(detections=self._make_detections())
        summary = dr.summary
        assert "person:97%" in summary
        assert "car:85%" in summary

    def test_summary_empty(self):
        dr = DetectionResult()
        assert dr.summary == ""

    def test_filter_by_pattern_keeps_matching(self):
        dr = DetectionResult(detections=self._make_detections(), frame_id=1)
        filtered = dr.filter_by_pattern("person")
        assert len(filtered.detections) == 1
        assert filtered.detections[0].label == "person"
        assert filtered.frame_id == 1

    def test_filter_by_pattern_removes_nonmatching(self):
        dr = DetectionResult(detections=self._make_detections())
        filtered = dr.filter_by_pattern("truck")
        assert len(filtered.detections) == 0

    def test_filter_by_pattern_regex(self):
        dr = DetectionResult(detections=self._make_detections())
        filtered = dr.filter_by_pattern("(person|car)")
        assert len(filtered.detections) == 2

    def test_to_dict(self):
        dets = self._make_detections()
        dr = DetectionResult(
            detections=dets,
            frame_id=5,
            image_dimensions={"original": (480, 640)},
        )
        d = dr.to_dict()
        assert d["labels"] == ["person", "car"]
        assert d["boxes"] == [[10, 20, 50, 80], [60, 30, 90, 70]]
        assert d["confidences"] == [0.97, 0.85]
        assert d["frame_id"] == 5
        assert d["image_dimensions"] == {"original": (480, 640)}
        assert d["model_names"] == ["yolov4", "yolov4"]
        assert d["error_boxes"] == []
        assert d["polygons"] == []

    def test_to_dict_with_error_boxes(self):
        eb = BBox(x1=0, y1=0, x2=10, y2=10)
        dr = DetectionResult(
            detections=self._make_detections(),
            error_boxes=[eb],
        )
        d = dr.to_dict()
        assert d["error_boxes"] == [[0, 0, 10, 10]]

    def test_annotate_no_image_raises(self):
        dr = DetectionResult(detections=self._make_detections())
        with pytest.raises(ValueError, match="No image"):
            dr.annotate()

    @pytest.mark.integration
    def test_annotate_with_mock_cv2(self):
        """Test annotate() by mocking cv2 to avoid requiring it."""
        mock_cv2 = MagicMock()
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.getTextSize.return_value = ((100, 20), 0)

        mock_np = MagicMock()
        mock_image = MagicMock()
        mock_image.copy.return_value = mock_image
        mock_image.shape = (100, 100, 3)

        dr = DetectionResult(
            detections=self._make_detections(),
            image=mock_image,
        )

        with patch.dict("sys.modules", {"cv2": mock_cv2, "numpy": mock_np}):
            result = dr.annotate()
            assert mock_cv2.rectangle.called
            assert mock_cv2.putText.called
