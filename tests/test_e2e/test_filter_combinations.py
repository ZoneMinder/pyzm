"""E2E: Combinations of filters applied together through the pipeline."""

from __future__ import annotations

from pathlib import Path

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model, load_image


class TestFilterCombinations:

    def test_pattern_plus_size_filter(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_combined = mc.model_copy(update={"pattern": ".*", "max_detection_size": "0.01%"})
        config = DetectorConfig(models=[mc_combined])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_zone_plus_pattern(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        from pyzm.models.zm import Zone
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc])
        det = Detector(config=config)
        img = load_image()
        h, w = img.shape[:2]
        zones = [Zone(name="z1", points=[(0, 0), (w, 0), (w, h), (0, h)], pattern="person")]
        result = det.detect(img, zones=zones)
        for d in result.detections:
            assert d.label == "person"

    def test_global_pattern_plus_zone(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        from pyzm.models.zm import Zone
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc], pattern="^zzz$")
        det = Detector(config=config)
        img = load_image()
        h, w = img.shape[:2]
        zones = [Zone(name="z1", points=[(0, 0), (w, 0), (w, h), (0, h)])]
        result = det.detect(img, zones=zones)
        assert len(result.detections) == 0

    def test_past_detection_plus_zone(self, tmp_path):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        from pyzm.models.zm import Zone
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        img = load_image()
        h, w = img.shape[:2]
        zones = [Zone(name="full", points=[(0, 0), (w, 0), (w, h), (0, h)])]
        config = DetectorConfig(
            models=[mc],
            match_past_detections=True,
            image_path=str(tmp_path),
        )
        det1 = Detector(config=config)
        result1 = det1.detect(img, zones=zones)
        first_count = len(result1.detections)

        det2 = Detector(config=config)
        result2 = det2.detect(img, zones=zones)
        if first_count > 0:
            assert len(result2.detections) < first_count or len(result2.detections) == 0
