"""E2E: Multi-model pipeline and multiple named models."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH


class TestMultiModelPipeline:

    def test_two_models_union(self):
        from pyzm.models.config import DetectorConfig, MatchStrategy, Processor
        from pyzm.ml.detector import Detector, _discover_models
        models = _discover_models(Path(BASE_PATH), Processor.CPU)
        if len(models) < 2:
            pytest.skip("Need at least 2 models for multi-model test")
        config = DetectorConfig(
            models=models[:2],
            match_strategy=MatchStrategy.UNION,
        )
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        if result.matched:
            model_names = {d.model_name for d in result.detections}
            assert len(model_names) >= 1

    def test_multiple_names_by_string(self):
        from pyzm.ml.detector import Detector, _discover_models
        from pyzm.models.config import Processor
        models = _discover_models(Path(BASE_PATH), Processor.CPU)
        names = [m.name for m in models[:3]]
        if len(names) < 1:
            pytest.skip("No models available")
        det = Detector(models=names, base_path=BASE_PATH, processor="cpu")
        assert len(det._config.models) == len(names)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)
