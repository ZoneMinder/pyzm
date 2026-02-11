"""AWS Rekognition backend adapter wrapping ``pyzm.ml.aws_rekognition``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class RekognitionBackend(MLBackend):
    """Wraps the legacy :class:`pyzm.ml.aws_rekognition.AwsRekognition` class."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: object | None = None

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "aws_rekognition"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from pyzm.ml.aws_rekognition import AwsRekognition  # lazy import

        options = self._config_to_legacy_options()
        logger.debug("%s: initializing AWS Rekognition client", self.name)
        self._model = AwsRekognition(options=options)

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
            "object_min_confidence": self._config.min_confidence,
            "aws_region": self._config.aws_region,
            "aws_access_key_id": self._config.aws_access_key_id,
            "aws_secret_access_key": self._config.aws_secret_access_key,
        }
