"""Coral EdgeTPU backend adapter wrapping ``pyzm.ml.coral_edgetpu.Tpu``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class CoralBackend(MLBackend):
    """Wraps the legacy :class:`pyzm.ml.coral_edgetpu.Tpu` class."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: object | None = None

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "coral_edgetpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from pyzm.ml.coral_edgetpu import Tpu  # lazy import

        options = self._config_to_legacy_options()
        logger.debug("%s: loading Coral EdgeTPU model", self.name)
        self._model = Tpu(options=options)
        self._model.load_model()

    def detect(self, image: "np.ndarray") -> list[Detection]:
        if self._model is None:
            self.load()

        assert self._model is not None
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
        return {
            "name": self.name,
            "object_weights": self._config.weights,
            "object_labels": self._config.labels,
            "object_min_confidence": self._config.min_confidence,
            "disable_locks": "yes" if self._config.disable_locks else "no",
            "auto_lock": not self._config.disable_locks,
            "tpu_max_processes": self._config.max_processes,
            "tpu_max_lock_wait": self._config.max_lock_wait,
        }
