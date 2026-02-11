"""ALPR backend adapter wrapping ``pyzm.ml.alpr.Alpr``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class AlprBackend(MLBackend):
    """Wraps the legacy :class:`pyzm.ml.alpr.Alpr` facade which delegates to
    PlateRecognizer, OpenALPR cloud, or OpenALPR command-line.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: object | None = None

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "alpr"

    @property
    def is_loaded(self) -> bool:
        # ALPR has no model to pre-load (API-based) - always "ready"
        return self._model is not None

    def load(self) -> None:
        from pyzm.ml.alpr import Alpr  # lazy import

        options = self._config_to_legacy_options()
        logger.debug("%s: initializing ALPR backend (%s)", self.name, self._config.alpr_service)
        self._model = Alpr(options=options)

    def detect(self, image: "np.ndarray") -> list[Detection]:
        if self._model is None:
            self.load()

        assert self._model is not None
        boxes, labels, confidences, _model_tags = self._model.detect(image=image)

        detections: list[Detection] = []
        for box, label, conf in zip(boxes, labels, confidences):
            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    model_name=self.name,
                    detection_type="alpr",
                )
            )
        return detections

    # -- internal helpers -----------------------------------------------------

    def _config_to_legacy_options(self) -> dict:
        return {
            "name": self.name,
            "alpr_service": self._config.alpr_service,
            "alpr_key": self._config.alpr_key,
            "alpr_url": self._config.alpr_url,
            "alpr_api_type": self._config.options.get("alpr_api_type", "cloud"),
            "platerec_min_dscore": self._config.platerec_min_dscore,
            "platerec_min_score": self._config.platerec_min_score,
            "platerec_stats": self._config.options.get("platerec_stats", "no"),
            "platerec_regions": self._config.options.get("platerec_regions"),
            "platerec_payload": self._config.options.get("platerec_payload"),
            "platerec_config": self._config.options.get("platerec_config"),
            "openalpr_country": self._config.options.get("openalpr_country"),
            "openalpr_state": self._config.options.get("openalpr_state"),
            "openalpr_recognize_vehicle": self._config.options.get("openalpr_recognize_vehicle"),
            "openalpr_cmdline_binary": self._config.options.get("openalpr_cmdline_binary"),
            "openalpr_cmdline_params": self._config.options.get("openalpr_cmdline_params", ""),
            "openalpr_cmdline_min_confidence": self._config.options.get(
                "openalpr_cmdline_min_confidence", 0.3
            ),
            "openalpr_min_confidence": self._config.options.get("openalpr_min_confidence", 0.3),
            "disable_locks": "yes" if self._config.disable_locks else "no",
            "max_size": self._config.max_detection_size,
            "resize": self._config.options.get("resize", "no"),
        }
