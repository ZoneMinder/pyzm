"""E2E: Frame strategies -- FIRST, MOST, MOST_UNIQUE, MOST_MODELS."""

from __future__ import annotations

from pathlib import Path

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model, load_image


class TestFrameStrategies:

    def _run_multi_frame(self, strategy):
        from pyzm.models.config import DetectorConfig, FrameStrategy, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(
            models=[mc],
            frame_strategy=FrameStrategy(strategy),
        )
        det = Detector(config=config)
        image = load_image()
        frames = [("f1", image), ("f2", image)]
        return det.detect(frames)

    def test_first_frame_strategy(self):
        result = self._run_multi_frame("first")
        assert result is not None
        if result.matched:
            assert result.frame_id == "f1"

    def test_most_frame_strategy(self):
        result = self._run_multi_frame("most")
        assert result is not None
        assert result.frame_id in ("f1", "f2")

    def test_most_unique_frame_strategy(self):
        result = self._run_multi_frame("most_unique")
        assert result is not None
        assert result.frame_id in ("f1", "f2")

    def test_most_models_frame_strategy(self):
        result = self._run_multi_frame("most_models")
        assert result is not None
        assert result.frame_id in ("f1", "f2")
