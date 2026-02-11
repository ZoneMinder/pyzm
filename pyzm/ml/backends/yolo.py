"""YOLO backend adapter wrapping the existing ``pyzm.ml.yolo`` code."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class YoloBackend(MLBackend):
    """Wraps the legacy :func:`pyzm.ml.yolo.Yolo` factory in the new backend
    interface.

    The underlying YOLO model (Darknet or ONNX) is chosen automatically based
    on the weights file extension.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: object | None = None

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "yolo"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from pyzm.ml.yolo import Yolo  # lazy import

        options = self._config_to_legacy_options()
        logger.debug("%s: loading YOLO model via legacy factory", self.name)
        self._model = Yolo(options=options)
        # Force model load so weights are in memory before first detect()
        self._model.load_model()

    def detect(self, image: "np.ndarray") -> list[Detection]:
        if self._model is None:
            self.load()

        assert self._model is not None
        # Legacy detect returns (boxes, labels, confidences, model_names)
        # boxes: list of [x1, y1, x2, y2]
        # labels: list of str
        # confidences: list of float
        # model_names: list of str (e.g. ["yolo", "yolo", ...])
        boxes, labels, confidences, _model_tags = self._model.detect(image=image)

        detections: list[Detection] = []
        for box, label, conf in zip(boxes, labels, confidences):
            if conf < self._config.min_confidence:
                logger.debug(
                    "%s: dropping %s (%.2f < %.2f)",
                    self.name, label, conf, self._config.min_confidence,
                )
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    model_name=self.name,
                    detection_type="object",
                )
            )
        return detections

    # -- internal helpers -----------------------------------------------------

    def _config_to_legacy_options(self) -> dict:
        """Translate a :class:`ModelConfig` into the old dict-of-strings the
        YOLO code expects."""
        opts: dict = {
            "name": self.name,
            "object_weights": self._config.weights,
            "object_config": self._config.config,
            "object_labels": self._config.labels,
            "object_min_confidence": self._config.min_confidence,
            "object_processor": self._config.processor.value,
            "disable_locks": "yes" if self._config.disable_locks else "no",
            "max_detection_size": self._config.max_detection_size,
            "auto_lock": not self._config.disable_locks,
            f"{self._config.processor.value}_max_processes": self._config.max_processes,
            f"{self._config.processor.value}_max_lock_wait": self._config.max_lock_wait,
        }
        # Only pass dimensions if explicitly set, so the YOLO/ONNX code can
        # use its own defaults (416 for Darknet, 640 for ONNX).
        if self._config.model_width is not None:
            opts["model_width"] = self._config.model_width
        if self._config.model_height is not None:
            opts["model_height"] = self._config.model_height
        return opts
