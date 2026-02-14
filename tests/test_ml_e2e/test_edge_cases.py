"""E2E: Edge cases and error handling."""

from __future__ import annotations

import pytest

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestEdgeCases:

    def test_empty_image_raises(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        with pytest.raises(FileNotFoundError):
            det.detect("/nonexistent/image.jpg")

    def test_empty_frames_list(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        pipeline = det._ensure_pipeline()
        result = det._detect_multi_frame([], None, pipeline)
        assert not result.matched

    def test_no_models_empty_result(self):
        from pyzm.models.config import DetectorConfig
        from pyzm.ml.detector import Detector
        import numpy as np
        config = DetectorConfig(models=[])
        det = Detector(config=config)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.detect(img)
        assert len(result.detections) == 0

    def test_detection_result_filter_by_pattern(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        result = det.detect(BIRD_IMAGE)
        filtered = result.filter_by_pattern("^zzz$")
        assert len(filtered.detections) == 0

    def test_annotate_returns_image(self):
        from pyzm.ml.detector import Detector
        import numpy as np
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        result = det.detect(BIRD_IMAGE)
        if not result.matched:
            pytest.skip("No detections to annotate")
        annotated = result.annotate()
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == result.image.shape

    def test_annotate_no_image_raises(self):
        from pyzm.models.detection import DetectionResult
        result = DetectionResult()
        with pytest.raises(ValueError, match="No image"):
            result.annotate()
