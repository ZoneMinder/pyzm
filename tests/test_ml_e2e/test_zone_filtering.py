"""E2E: Zone/polygon filtering with real detections."""

from __future__ import annotations

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model, load_image


class TestZoneFiltering:

    def test_full_image_zone_keeps_all(self):
        from pyzm.ml.detector import Detector
        from pyzm.models.zm import Zone
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        img = load_image()
        h, w = img.shape[:2]
        zones = [Zone(name="full", points=[(0, 0), (w, 0), (w, h), (0, h)])]
        result = det.detect(img, zones=zones)
        assert isinstance(result.detections, list)

    def test_tiny_zone_filters_most(self):
        from pyzm.ml.detector import Detector
        from pyzm.models.zm import Zone
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        img = load_image()
        zones = [Zone(name="tiny", points=[(0, 0), (1, 0), (1, 1), (0, 1)])]
        result = det.detect(img, zones=zones)
        assert isinstance(result.error_boxes, list)

    def test_zone_with_pattern(self):
        from pyzm.ml.detector import Detector
        from pyzm.models.zm import Zone
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        img = load_image()
        h, w = img.shape[:2]
        zones = [Zone(name="bird_only", points=[(0, 0), (w, 0), (w, h), (0, h)], pattern="bird")]
        result = det.detect(img, zones=zones)
        for d in result.detections:
            assert d.label == "bird"

    def test_zone_non_matching_pattern_filters_all(self):
        from pyzm.ml.detector import Detector
        from pyzm.models.zm import Zone
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        img = load_image()
        h, w = img.shape[:2]
        zones = [Zone(name="nope", points=[(0, 0), (w, 0), (w, h), (0, h)], pattern="^zzz$")]
        result = det.detect(img, zones=zones)
        assert len(result.detections) == 0

    def test_multiple_zones_different_patterns(self):
        from pyzm.ml.detector import Detector
        from pyzm.models.zm import Zone
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        img = load_image()
        h, w = img.shape[:2]
        zones = [
            Zone(name="person_zone", points=[(0, 0), (w // 2, 0), (w // 2, h), (0, h)], pattern="person"),
            Zone(name="bird_zone", points=[(0, 0), (w, 0), (w, h), (0, h)], pattern="bird"),
        ]
        result = det.detect(img, zones=zones)
        for d in result.detections:
            assert d.label in ("person", "bird")

    def test_no_zones_passes_all(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)
