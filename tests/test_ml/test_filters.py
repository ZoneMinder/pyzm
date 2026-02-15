"""Tests for pyzm.ml.filters -- detection filtering functions."""

from __future__ import annotations

import os
import pickle
from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.detection import BBox, Detection


# ===================================================================
# Helpers
# ===================================================================

def _det(label: str, x1: int, y1: int, x2: int, y2: int, conf: float = 0.9) -> Detection:
    """Shortcut to create a Detection."""
    return Detection(
        label=label,
        confidence=conf,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        model_name="test",
    )


# ===================================================================
# TestFilterByZone
# ===================================================================

@pytest.mark.integration
class TestFilterByZone:
    """Tests for filter_by_zone. Requires shapely."""

    def test_detection_inside_zone_passes(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]
        zones = [{"name": "zone1", "points": [(0, 0), (100, 0), (100, 100), (0, 100)]}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"
        assert len(error_boxes) == 0

    def test_detection_outside_zone_filtered(self):
        from pyzm.ml.filters import filter_by_zone

        # Detection is at (200, 200) - (250, 250), zone is at (0,0)-(100,100)
        dets = [_det("person", 200, 200, 250, 250)]
        zones = [{"name": "zone1", "points": [(0, 0), (100, 0), (100, 100), (0, 100)]}]

        kept, error_boxes = filter_by_zone(dets, zones, (300, 300))
        assert len(kept) == 0
        assert len(error_boxes) == 1

    def test_zone_pattern_filters_labels(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [
            _det("person", 10, 10, 40, 40),
            _det("car", 10, 10, 40, 40),
        ]
        zones = [{"name": "driveway", "points": [(0, 0), (100, 0), (100, 100), (0, 100)], "pattern": "person"}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"
        assert len(error_boxes) == 1

    def test_no_zones_synthesises_full_image(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]

        kept, error_boxes = filter_by_zone(dets, [], (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"

    def test_multiple_zones_first_match_wins(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("dog", 10, 10, 40, 40)]
        zones = [
            {"name": "zone1", "points": [(0, 0), (50, 0), (50, 50), (0, 50)], "pattern": "person"},
            {"name": "zone2", "points": [(0, 0), (50, 0), (50, 50), (0, 50)], "pattern": "dog"},
        ]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "dog"

    def test_zone_with_value_key(self):
        """Test backward compat: zone dict uses 'value' instead of 'points'."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]
        zones = [{"name": "zone1", "value": [(0, 0), (100, 0), (100, 100), (0, 100)]}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1

    def test_zone_pattern_none_matches_all(self):
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("anything", 10, 10, 40, 40)]
        zones = [{"name": "zone1", "points": [(0, 0), (100, 0), (100, 100), (0, 100)], "pattern": None}]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1

    # -- ignore_pattern tests (Ref: ZoneMinder/pyzm#37) --

    def test_ignore_pattern_suppresses_matching_label(self):
        """Detection matching ignore_pattern in an intersecting zone is suppressed."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", 10, 10, 40, 40)]
        zones = [{
            "name": "driveway",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ignore_pattern": "(car|truck)",
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 0
        assert len(error_boxes) == 1

    def test_ignore_pattern_does_not_suppress_non_matching(self):
        """Detection NOT matching ignore_pattern passes through normally."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("person", 10, 10, 40, 40)]
        zones = [{
            "name": "driveway",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ignore_pattern": "(car|truck)",
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"

    def test_ignore_pattern_with_positive_pattern(self):
        """ignore_pattern takes precedence over positive pattern for matching labels."""
        from pyzm.ml.filters import filter_by_zone

        dets = [
            _det("car", 10, 10, 40, 40),
            _det("person", 10, 10, 40, 40),
        ]
        zones = [{
            "name": "driveway",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "pattern": "(person|car)",
            "ignore_pattern": "car",
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1
        assert kept[0].label == "person"

    def test_ignore_pattern_none_does_not_suppress(self):
        """When ignore_pattern is None, no suppression occurs."""
        from pyzm.ml.filters import filter_by_zone

        dets = [_det("car", 10, 10, 40, 40)]
        zones = [{
            "name": "zone1",
            "points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ignore_pattern": None,
        }]

        kept, error_boxes = filter_by_zone(dets, zones, (100, 100))
        assert len(kept) == 1


# ===================================================================
# TestFilterBySize
# ===================================================================

class TestFilterBySize:
    """Tests for filter_by_size."""

    def test_filter_by_size_percentage(self):
        from pyzm.ml.filters import filter_by_size

        # Image is 100x100 = 10000px area. Detection is 50x50 = 2500px.
        # 50% of 10000 = 5000. Detection area (2500) < 5000 -> kept.
        dets = [_det("person", 0, 0, 50, 50)]
        result = filter_by_size(dets, "50%", (100, 100))
        assert len(result) == 1

    def test_filter_by_size_percentage_too_large(self):
        from pyzm.ml.filters import filter_by_size

        # 90x90 = 8100px area. 50% of 10000 = 5000. 8100 > 5000 -> filtered.
        dets = [_det("person", 0, 0, 90, 90)]
        result = filter_by_size(dets, "50%", (100, 100))
        assert len(result) == 0

    def test_filter_by_size_pixels(self):
        from pyzm.ml.filters import filter_by_size

        # Detection is 50x50 = 2500px. 300px threshold -> 2500 > 300 -> filtered.
        dets = [_det("person", 0, 0, 50, 50)]
        result = filter_by_size(dets, "300px", (100, 100))
        assert len(result) == 0

    def test_filter_by_size_pixels_passes(self):
        from pyzm.ml.filters import filter_by_size

        # Detection is 5x5 = 25px. 300px threshold -> 25 < 300 -> kept.
        dets = [_det("person", 0, 0, 5, 5)]
        result = filter_by_size(dets, "300px", (100, 100))
        assert len(result) == 1

    def test_filter_by_size_none_passes_all(self):
        from pyzm.ml.filters import filter_by_size

        dets = [_det("person", 0, 0, 99, 99)]
        result = filter_by_size(dets, None, (100, 100))
        assert len(result) == 1

    def test_filter_by_size_empty_string_passes_all(self):
        from pyzm.ml.filters import filter_by_size

        dets = [_det("person", 0, 0, 99, 99)]
        result = filter_by_size(dets, "", (100, 100))
        assert len(result) == 1

    def test_filter_by_size_multiple_detections(self):
        from pyzm.ml.filters import filter_by_size

        dets = [
            _det("small", 0, 0, 10, 10),   # area = 100
            _det("large", 0, 0, 80, 80),    # area = 6400
        ]
        result = filter_by_size(dets, "1000px", (100, 100))
        assert len(result) == 1
        assert result[0].label == "small"


# ===================================================================
# TestFilterByPattern
# ===================================================================

class TestFilterByPattern:
    """Tests for filter_by_pattern."""

    def test_regex_matching(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [
            _det("person", 0, 0, 10, 10),
            _det("car", 10, 10, 20, 20),
            _det("dog", 20, 20, 30, 30),
        ]
        result = filter_by_pattern(dets, "(person|car)")
        assert len(result) == 2
        assert result[0].label == "person"
        assert result[1].label == "car"

    def test_wildcard_matches_all(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10), _det("car", 10, 10, 20, 20)]
        result = filter_by_pattern(dets, ".*")
        assert len(result) == 2

    def test_empty_pattern_matches_all(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10)]
        result = filter_by_pattern(dets, "")
        assert len(result) == 1

    def test_no_match(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10)]
        result = filter_by_pattern(dets, "truck")
        assert len(result) == 0

    def test_partial_match_with_prefix(self):
        from pyzm.ml.filters import filter_by_pattern

        dets = [_det("person", 0, 0, 10, 10)]
        result = filter_by_pattern(dets, "per.*")
        assert len(result) == 1


# ===================================================================
# TestFilterPastDetections
# ===================================================================

@pytest.mark.integration
class TestFilterPastDetections:
    """Tests for filter_past_detections. Requires shapely + pickle."""

    def test_no_past_file_returns_all(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        dets = [_det("person", 10, 10, 50, 50)]
        past_file = str(tmp_path / "past.pkl")

        result = filter_past_detections(dets, past_file, "5%")
        assert len(result) == 1

    def test_saves_detections(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        dets = [_det("person", 10, 10, 50, 50)]
        past_file = str(tmp_path / "past.pkl")

        filter_past_detections(dets, past_file, "5%")

        # Verify the file was created
        assert os.path.exists(past_file)

        # Verify contents
        with open(past_file, "rb") as fh:
            saved_boxes = pickle.load(fh)
            saved_labels = pickle.load(fh)

        assert len(saved_boxes) == 1
        assert saved_boxes[0] == [10, 10, 50, 50]
        assert saved_labels[0] == "person"

    def test_duplicate_detection_removed(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        past_file = str(tmp_path / "past.pkl")

        # Save initial detection
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["person"], fh)

        # Same detection (same label, same bbox) should be filtered
        dets = [_det("person", 10, 10, 50, 50)]
        result = filter_past_detections(dets, past_file, "5%")
        assert len(result) == 0

    def test_different_label_not_filtered(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        past_file = str(tmp_path / "past.pkl")

        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["person"], fh)

        # Different label -> should NOT be filtered
        dets = [_det("car", 10, 10, 50, 50)]
        result = filter_past_detections(dets, past_file, "5%")
        assert len(result) == 1

    def test_moved_detection_passes(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        past_file = str(tmp_path / "past.pkl")

        # Save detection at (10, 10, 50, 50)
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["person"], fh)

        # Detection at a very different location -> should pass
        dets = [_det("person", 200, 200, 300, 300)]
        result = filter_past_detections(dets, past_file, "5%")
        assert len(result) == 1

    def test_empty_past_file_handled(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        past_file = str(tmp_path / "past.pkl")
        # Create an empty file
        with open(past_file, "wb") as fh:
            pass

        dets = [_det("person", 10, 10, 50, 50)]
        result = filter_past_detections(dets, past_file, "5%")
        assert len(result) == 1

    def test_aliases_match_across_labels(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        past_file = str(tmp_path / "past.pkl")
        # Save a "bus" detection
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["bus"], fh)

        # "car" at the same location, aliased with bus -> should be filtered
        dets = [_det("car", 10, 10, 50, 50)]
        result = filter_past_detections(
            dets, past_file, "5%", aliases=[["car", "bus", "truck"]],
        )
        assert len(result) == 0

    def test_ignore_labels_always_kept(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        past_file = str(tmp_path / "past.pkl")
        # Save a "dog" detection at same spot
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["dog"], fh)

        # "dog" at the same location but ignored -> should be kept
        dets = [_det("dog", 10, 10, 50, 50)]
        result = filter_past_detections(
            dets, past_file, "5%", ignore_labels=["dog"],
        )
        assert len(result) == 1

    def test_per_label_area_override(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections

        past_file = str(tmp_path / "past.pkl")
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["car"], fh)

        # Car at slightly shifted position — with default 5% would be filtered,
        # but with a very strict 0% override it should pass
        dets = [_det("car", 11, 11, 51, 51)]
        result = filter_past_detections(
            dets, past_file, "50%", label_area_overrides={"car": "0px"},
        )
        assert len(result) == 1


# ===================================================================
# TestLoadSavePastDetections
# ===================================================================

class TestLoadSavePastDetections:
    """Tests for the composable load/save helpers."""

    def test_round_trip(self, tmp_path):
        from pyzm.ml.filters import load_past_detections, save_past_detections

        past_file = str(tmp_path / "past.pkl")
        dets = [_det("person", 10, 10, 50, 50), _det("car", 60, 60, 100, 100)]
        save_past_detections(past_file, dets)

        boxes, labels = load_past_detections(past_file)
        assert len(boxes) == 2
        assert boxes[0] == [10, 10, 50, 50]
        assert boxes[1] == [60, 60, 100, 100]
        assert labels == ["person", "car"]

    def test_load_missing_file(self, tmp_path):
        from pyzm.ml.filters import load_past_detections

        boxes, labels = load_past_detections(str(tmp_path / "nonexistent.pkl"))
        assert boxes == []
        assert labels == []

    def test_load_empty_file(self, tmp_path):
        from pyzm.ml.filters import load_past_detections

        past_file = str(tmp_path / "empty.pkl")
        with open(past_file, "wb"):
            pass
        boxes, labels = load_past_detections(past_file)
        assert boxes == []
        assert labels == []

    def test_save_empty_detections_is_noop(self, tmp_path):
        from pyzm.ml.filters import save_past_detections

        past_file = str(tmp_path / "past.pkl")
        save_past_detections(past_file, [])
        assert not os.path.exists(past_file)


# ===================================================================
# TestMatchPastDetectionsPure
# ===================================================================

@pytest.mark.integration
class TestMatchPastDetectionsPure:
    """Tests for the pure match_past_detections logic (no I/O)."""

    def test_no_saved_data_returns_all(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("person", 10, 10, 50, 50)]
        result = match_past_detections(dets, [], [], "5%")
        assert len(result) == 1

    def test_duplicate_removed(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("person", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["person"], "5%",
        )
        assert len(result) == 0

    def test_different_label_kept(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("car", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["person"], "5%",
        )
        assert len(result) == 1

    def test_aliases_match_across_labels(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("car", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["bus"], "5%",
            aliases=[["car", "bus"]],
        )
        assert len(result) == 0

    def test_ignore_labels_always_kept(self):
        from pyzm.ml.filters import match_past_detections

        dets = [_det("dog", 10, 10, 50, 50)]
        result = match_past_detections(
            dets, [[10, 10, 50, 50]], ["dog"], "5%",
            ignore_labels=["dog"],
        )
        assert len(result) == 1
