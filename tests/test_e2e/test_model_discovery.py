"""E2E: Model auto-discovery and name resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_e2e.conftest import BASE_PATH


class TestModelDiscovery:

    def test_auto_discover_all(self):
        from pyzm.ml.detector import Detector
        det = Detector(models=None, base_path=BASE_PATH, processor="cpu")
        assert len(det._config.models) > 0

    def test_resolve_by_directory_name(self):
        from pyzm.ml.detector import _resolve_model_name
        from pyzm.models.config import Processor
        if not (Path(BASE_PATH) / "yolov4").is_dir():
            pytest.skip("yolov4 directory not found")
        mc = _resolve_model_name("yolov4", Path(BASE_PATH), Processor.CPU)
        assert mc.name is not None
        assert mc.weights is not None

    def test_resolve_by_file_stem(self):
        from pyzm.ml.detector import _resolve_model_name
        from pyzm.models.config import Processor
        bp = Path(BASE_PATH)
        onnx_found = None
        for subdir in bp.iterdir():
            if subdir.is_dir():
                for f in subdir.iterdir():
                    if f.suffix == ".onnx":
                        onnx_found = f.stem
                        break
            if onnx_found:
                break
        if not onnx_found:
            pytest.skip("No ONNX model found for file stem test")
        mc = _resolve_model_name(onnx_found, bp, Processor.CPU)
        assert mc.weights is not None
        assert mc.name == onnx_found

    def test_unknown_model_creates_bare_config(self):
        from pyzm.ml.detector import _resolve_model_name
        from pyzm.models.config import Processor
        mc = _resolve_model_name("nonexistent_model_xyz", Path(BASE_PATH), Processor.GPU)
        assert mc.name == "nonexistent_model_xyz"
        assert mc.processor.value == "gpu"
        assert mc.weights is None

    def test_discover_assigns_correct_framework(self):
        from pyzm.ml.detector import _discover_models
        from pyzm.models.config import ModelFramework, Processor
        models = _discover_models(Path(BASE_PATH), Processor.CPU)
        for mc in models:
            if mc.weights and mc.weights.endswith(".onnx"):
                assert mc.framework == ModelFramework.OPENCV
            elif mc.weights and mc.weights.endswith(".weights"):
                assert mc.framework == ModelFramework.OPENCV
            elif mc.weights and mc.weights.endswith(".tflite"):
                assert mc.framework == ModelFramework.CORAL
