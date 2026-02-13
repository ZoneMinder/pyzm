"""E2E: Pipeline lazy (prepare) vs eager (load) loading."""

from __future__ import annotations

from pathlib import Path

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestPipelineLazyEager:

    def test_eager_load_marks_backends_loaded(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc])
        det = Detector(config=config)
        pipeline = det._ensure_pipeline(lazy=False)
        for _, backend in pipeline._backends:
            assert backend.is_loaded

    def test_lazy_prepare_no_weights_loaded(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc])
        det = Detector(config=config)
        pipeline = det._ensure_pipeline(lazy=True)
        for _, backend in pipeline._backends:
            assert not backend.is_loaded

    def test_lazy_loads_on_detect(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc])
        det = Detector(config=config)
        det._ensure_pipeline(lazy=True)
        for _, backend in det._pipeline._backends:
            assert not backend.is_loaded
        det.detect(BIRD_IMAGE)
        for _, backend in det._pipeline._backends:
            assert backend.is_loaded
