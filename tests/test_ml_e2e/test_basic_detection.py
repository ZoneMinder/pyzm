"""E2E: Basic detection -- single model, single image."""

from __future__ import annotations

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model, load_image


class TestBasicDetection:

    def test_detect_from_file_path(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        result = det.detect(BIRD_IMAGE)
        assert result is not None
        assert result.frame_id == "single"
        assert isinstance(result.labels, list)

    def test_detect_from_numpy_array(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        image = load_image()
        result = det.detect(image)
        assert result is not None
        assert isinstance(result.labels, list)

    def test_detect_multi_frame(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        image = load_image()
        frames = [("frame1", image), ("frame2", image)]
        result = det.detect(frames)
        assert result is not None
        assert result.frame_id in ("frame1", "frame2")

    def test_detection_result_properties(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.labels, list)
        assert isinstance(result.confidences, list)
        assert isinstance(result.boxes, list)
        assert isinstance(result.summary, str)
        assert isinstance(result.matched, bool)
        assert len(result.labels) == len(result.confidences) == len(result.boxes)

    def test_detection_result_to_dict_roundtrip(self):
        from pyzm.ml.detector import Detector
        from pyzm.models.detection import DetectionResult
        model = find_one_model()
        det = Detector(models=[model], base_path=BASE_PATH, processor="cpu")
        result = det.detect(BIRD_IMAGE)
        d = result.to_dict()
        reconstructed = DetectionResult.from_dict(d)
        assert reconstructed.labels == result.labels
        assert len(reconstructed.boxes) == len(result.boxes)
