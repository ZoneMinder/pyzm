"""Tests for pyzm.ml.detector -- top-level Detector API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import (
    DetectorConfig,
    FrameStrategy,
    MatchStrategy,
    ModelConfig,
    ModelFramework,
    ModelType,
    Processor,
)
from pyzm.models.detection import BBox, Detection, DetectionResult


# ===================================================================
# Helpers
# ===================================================================

def _det(label: str, conf: float = 0.9) -> Detection:
    return Detection(
        label=label,
        confidence=conf,
        bbox=BBox(x1=10, y1=10, x2=50, y2=50),
        model_name="test",
    )


def _make_pipeline_mock(detections: list[Detection] | None = None) -> MagicMock:
    """Create a mock ModelPipeline."""
    pipeline = MagicMock()
    result = DetectionResult(detections=detections or [])
    pipeline.run.return_value = result
    return pipeline


# ===================================================================
# TestDetectorInit
# ===================================================================

class TestDetectorInit:
    """Tests for Detector.__init__ with various argument types."""

    def test_init_with_model_name_strings(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])
        assert len(det._config.models) == 1
        assert det._config.models[0].name == "yolov4"
        assert det._config.models[0].framework == ModelFramework.OPENCV
        assert det._config.models[0].type == ModelType.OBJECT

    def test_init_with_multiple_model_names(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4", "face_dlib"])
        assert len(det._config.models) == 2
        assert det._config.models[0].name == "yolov4"
        assert det._config.models[1].name == "face_dlib"
        assert det._config.models[1].type == ModelType.FACE
        assert det._config.models[1].framework == ModelFramework.FACE_DLIB

    def test_init_with_model_config_objects(self):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(
            name="custom-model",
            type=ModelType.OBJECT,
            framework=ModelFramework.OPENCV,
            weights="/path/to/weights",
        )
        det = Detector(models=[mc])
        assert len(det._config.models) == 1
        assert det._config.models[0] is mc

    def test_init_with_mixed_strings_and_configs(self):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(name="custom", type=ModelType.FACE)
        det = Detector(models=["yolov4", mc])
        assert len(det._config.models) == 2
        assert det._config.models[0].name == "yolov4"
        assert det._config.models[1].name == "custom"

    def test_init_with_detector_config(self):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(name="yolov7", framework=ModelFramework.OPENCV)
        config = DetectorConfig(
            models=[mc],
            match_strategy=MatchStrategy.MOST,
            frame_strategy=FrameStrategy.MOST,
        )
        det = Detector(config=config)
        assert det._config is config
        assert det._config.match_strategy == MatchStrategy.MOST

    def test_init_with_no_args_empty_base(self):
        from pyzm.ml.detector import Detector

        det = Detector(base_path="/nonexistent/path")
        assert det._config.models == []

    def test_init_unknown_preset_defaults_to_yolov4(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["nonexistent_model"])
        mc = det._config.models[0]
        assert mc.name == "nonexistent_model"
        # Should have defaulted to yolov4 preset
        assert mc.framework == ModelFramework.OPENCV
        assert mc.type == ModelType.OBJECT

    def test_coral_preset(self):
        from pyzm.ml.detector import Detector

        det = Detector(models=["coral"])
        mc = det._config.models[0]
        assert mc.framework == ModelFramework.CORAL
        assert mc.processor == Processor.TPU


# ===================================================================
# TestDetectorDetect
# ===================================================================

class TestDetectorDetect:
    """Tests for Detector.detect() method."""

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_single_image_mock(self, mock_pipeline_cls):
        """Test detect() with a single numpy array (mock).

        The detect() method does ``import numpy as np`` lazily and then
        ``isinstance(input, np.ndarray)``.  We mock numpy at the
        sys.modules level so the isinstance check works with our mock
        image.
        """
        pipeline = _make_pipeline_mock([_det("person"), _det("car")])
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])

        # Create a proper mock numpy module with an ndarray class
        import types
        mock_np_module = types.ModuleType("numpy")
        mock_image = MagicMock()

        # Make isinstance(mock_image, mock_np_module.ndarray) return True
        class FakeNdarray:
            pass

        mock_np_module.ndarray = FakeNdarray  # type: ignore[attr-defined]
        mock_image.__class__ = FakeNdarray
        mock_image.shape = (100, 100, 3)

        with patch.dict("sys.modules", {"numpy": mock_np_module}):
            result = det.detect(mock_image)

        assert isinstance(result, DetectionResult)
        assert result.frame_id == "single"

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_multiple_frames_most_strategy(self, mock_pipeline_cls):
        """Test detect() with multiple frames and MOST frame strategy."""
        # Frame 1: 1 detection, Frame 2: 3 detections
        result1 = DetectionResult(detections=[_det("person")])
        result2 = DetectionResult(detections=[_det("person"), _det("car"), _det("dog")])

        pipeline = MagicMock()
        pipeline.run.side_effect = [result1, result2]
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        config = DetectorConfig(
            models=[ModelConfig(name="test")],
            frame_strategy=FrameStrategy.MOST,
        )
        det = Detector(config=config)

        # Create mock frames
        mock_img1 = MagicMock()
        mock_img2 = MagicMock()
        frames = [(1, mock_img1), (2, mock_img2)]

        result = det.detect(frames)

        # Should pick frame 2 (3 detections > 1)
        assert len(result.detections) == 3
        assert result.frame_id == 2

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_string_path(self, mock_pipeline_cls):
        """Test detect() with a file path."""
        pipeline = _make_pipeline_mock([_det("person")])
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])

        # Mock cv2.imread
        mock_cv2 = MagicMock()
        mock_image = MagicMock()
        mock_cv2.imread.return_value = mock_image

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = det.detect("/path/to/image.jpg")

        assert result.frame_id == "single"

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_string_path_not_found(self, mock_pipeline_cls):
        """Test detect() raises FileNotFoundError for bad path."""
        mock_pipeline_cls.return_value = MagicMock()

        from pyzm.ml.detector import Detector

        det = Detector(models=["yolov4"])

        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = None  # cv2 returns None for bad files

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            with pytest.raises(FileNotFoundError):
                det.detect("/nonexistent/image.jpg")

    @patch("pyzm.ml.detector.ModelPipeline")
    def test_detect_multiple_frames_first_strategy(self, mock_pipeline_cls):
        """Test FIRST frame strategy: returns first matched frame."""
        result1 = DetectionResult(detections=[_det("person")])
        result2 = DetectionResult(detections=[_det("person"), _det("car")])

        pipeline = MagicMock()
        pipeline.run.side_effect = [result1, result2]
        mock_pipeline_cls.return_value = pipeline

        from pyzm.ml.detector import Detector

        config = DetectorConfig(
            models=[ModelConfig(name="test")],
            frame_strategy=FrameStrategy.FIRST,
        )
        det = Detector(config=config)

        frames = [(1, MagicMock()), (2, MagicMock())]
        result = det.detect(frames)

        # Should return first matched frame
        assert result.frame_id == 1
        assert len(result.detections) == 1


# ===================================================================
# TestModelDiscovery
# ===================================================================

class TestModelDiscovery:
    """Tests for base_path auto-discovery and name resolution."""

    def _make_darknet_dir(self, base, name, weights_name=None, cfg_name=None, labels_name="coco.names"):
        """Create a directory with Darknet-style model files."""
        d = base / name
        d.mkdir()
        (d / (weights_name or f"{name}.weights")).touch()
        if cfg_name is not False:
            (d / (cfg_name or f"{name}.cfg")).touch()
        (d / labels_name).touch()
        return d

    def _make_onnx_dir(self, base, dirname, *model_names):
        """Create a directory with ONNX model files."""
        d = base / dirname
        d.mkdir()
        for n in model_names:
            (d / f"{n}.onnx").touch()
        return d

    def test_discover_darknet_models(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        self._make_darknet_dir(tmp_path, "yolov4")
        models = _discover_models(tmp_path)

        assert len(models) == 1
        assert models[0].name == "yolov4"
        assert models[0].weights.endswith("yolov4.weights")
        assert models[0].config.endswith("yolov4.cfg")
        assert models[0].labels.endswith("coco.names")
        assert models[0].framework == ModelFramework.OPENCV

    def test_discover_onnx_models(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26n", "yolo26s")
        models = _discover_models(tmp_path)

        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"yolo26n", "yolo26s"}
        for m in models:
            assert m.weights.endswith(".onnx")
            assert m.config is None
            assert m.labels is None

    def test_discover_tflite_models(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        d = tmp_path / "coral"
        d.mkdir()
        (d / "ssd_mobilenet.tflite").touch()
        (d / "coco_labels.txt").touch()

        models = _discover_models(d.parent)
        coral_models = [m for m in models if m.name == "ssd_mobilenet"]
        assert len(coral_models) == 1
        assert coral_models[0].framework == ModelFramework.CORAL
        assert coral_models[0].processor == Processor.TPU
        assert coral_models[0].labels.endswith("coco_labels.txt")

    def test_discover_mixed_dirs(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        self._make_darknet_dir(tmp_path, "yolov4")
        self._make_onnx_dir(tmp_path, "ultralytics", "yolo11n", "yolo26s")

        models = _discover_models(tmp_path)
        assert len(models) == 3
        names = {m.name for m in models}
        assert "yolov4" in names
        assert "yolo11n" in names
        assert "yolo26s" in names

    def test_discover_empty_dir(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        models = _discover_models(tmp_path)
        assert models == []

    def test_discover_nonexistent_dir(self, tmp_path):
        from pyzm.ml.detector import _discover_models

        models = _discover_models(tmp_path / "nonexistent")
        assert models == []

    def test_resolve_by_directory_name(self, tmp_path):
        from pyzm.ml.detector import _resolve_model_name

        self._make_darknet_dir(tmp_path, "yolov4")
        mc = _resolve_model_name("yolov4", tmp_path)

        assert mc.name == "yolov4"
        assert mc.weights.endswith("yolov4.weights")
        assert mc.config.endswith("yolov4.cfg")

    def test_resolve_by_file_stem(self, tmp_path):
        from pyzm.ml.detector import _resolve_model_name

        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26s", "yolo26n")
        mc = _resolve_model_name("yolo26s", tmp_path)

        assert mc.name == "yolo26s"
        assert mc.weights.endswith("yolo26s.onnx")

    def test_resolve_falls_back_to_preset(self, tmp_path):
        from pyzm.ml.detector import _resolve_model_name

        mc = _resolve_model_name("face_dlib", tmp_path)
        assert mc.name == "face_dlib"
        assert mc.type == ModelType.FACE
        assert mc.weights is None  # no files found

    def test_detector_init_with_base_path(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        det = Detector(models=["yolov4"], base_path=tmp_path)

        assert len(det._config.models) == 1
        assert det._config.models[0].weights.endswith("yolov4.weights")

    def test_detector_init_auto_discover(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26s")

        det = Detector(base_path=tmp_path)
        assert len(det._config.models) == 2

    def test_detector_init_auto_discover_empty(self, tmp_path):
        from pyzm.ml.detector import Detector

        det = Detector(base_path=tmp_path)
        assert det._config.models == []

    def test_processor_string_gpu(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        det = Detector(models=["yolov4"], base_path=tmp_path, processor="gpu")

        assert det._config.models[0].processor == Processor.GPU

    def test_processor_enum_gpu(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        det = Detector(models=["yolov4"], base_path=tmp_path, processor=Processor.GPU)

        assert det._config.models[0].processor == Processor.GPU

    def test_processor_applies_to_auto_discover(self, tmp_path):
        from pyzm.ml.detector import Detector

        self._make_darknet_dir(tmp_path, "yolov4")
        self._make_onnx_dir(tmp_path, "ultralytics", "yolo26s")

        det = Detector(base_path=tmp_path, processor="gpu")
        for m in det._config.models:
            assert m.processor == Processor.GPU

    def test_processor_does_not_override_tflite(self, tmp_path):
        from pyzm.ml.detector import Detector

        d = tmp_path / "coral"
        d.mkdir()
        (d / "model.tflite").touch()
        (d / "labels.txt").touch()

        det = Detector(base_path=tmp_path, processor="gpu")
        tflite_models = [m for m in det._config.models if m.name == "model"]
        assert tflite_models[0].processor == Processor.TPU

    def test_processor_not_applied_to_model_config_objects(self, tmp_path):
        from pyzm.ml.detector import Detector

        mc = ModelConfig(name="custom", type=ModelType.OBJECT, processor=Processor.CPU)
        det = Detector(models=[mc], base_path=tmp_path, processor="gpu")

        assert det._config.models[0].processor == Processor.CPU


# ===================================================================
# TestDetectorFromDict
# ===================================================================

class TestDetectorFromDict:
    """Tests for Detector.from_dict() class method."""

    def test_from_dict(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {
                "model_sequence": "object",
                "same_model_sequence_strategy": "first",
                "frame_strategy": "most_models",
            },
            "object": {
                "general": {"pattern": "(person|car)"},
                "sequence": [
                    {
                        "name": "YOLOv4",
                        "object_framework": "opencv",
                        "object_processor": "cpu",
                        "object_weights": "/models/yolov4.weights",
                        "object_config": "/models/yolov4.cfg",
                        "object_labels": "/models/coco.names",
                        "object_min_confidence": "0.3",
                    }
                ],
            },
        }

        det = Detector.from_dict(ml_options)
        assert len(det._config.models) == 1
        assert det._config.models[0].name == "YOLOv4"
        assert det._config.match_strategy == MatchStrategy.FIRST
        assert det._config.frame_strategy == FrameStrategy.MOST_MODELS

    def test_from_dict_multi_model(self):
        from pyzm.ml.detector import Detector

        ml_options = {
            "general": {
                "model_sequence": "object,face",
                "same_model_sequence_strategy": "first",
            },
            "object": {
                "general": {},
                "sequence": [
                    {"name": "YOLO", "object_framework": "opencv"},
                ],
            },
            "face": {
                "general": {"pre_existing_labels": ["person"]},
                "sequence": [
                    {"name": "dlib", "face_detection_framework": "dlib"},
                ],
            },
        }

        det = Detector.from_dict(ml_options)
        assert len(det._config.models) == 2
        assert det._config.models[0].type == ModelType.OBJECT
        assert det._config.models[1].type == ModelType.FACE


# ===================================================================
# TestIsBetter
# ===================================================================

class TestIsBetter:
    """Tests for the _is_better helper function."""

    def test_most_strategy(self):
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person"), _det("car")])
        r2 = DetectionResult(detections=[_det("person")])
        assert _is_better(r1, r2, FrameStrategy.MOST) is True
        assert _is_better(r2, r1, FrameStrategy.MOST) is False

    def test_most_unique_strategy(self):
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person"), _det("car")])
        r2 = DetectionResult(detections=[_det("person"), _det("person")])
        assert _is_better(r1, r2, FrameStrategy.MOST_UNIQUE) is True

    def test_most_models_strategy(self):
        from pyzm.ml.detector import _is_better

        d1 = Detection(label="person", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="model_a")
        d2 = Detection(label="car", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="model_b")
        d3 = Detection(label="person", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="model_a")

        r1 = DetectionResult(detections=[d1, d2])  # 2 models
        r2 = DetectionResult(detections=[d3])       # 1 model
        assert _is_better(r1, r2, FrameStrategy.MOST_MODELS) is True

    def test_first_strategy(self):
        from pyzm.ml.detector import _is_better

        r1 = DetectionResult(detections=[_det("person")])
        r2 = DetectionResult(detections=[])
        assert _is_better(r1, r2, FrameStrategy.FIRST) is True
        assert _is_better(r2, r1, FrameStrategy.FIRST) is False
