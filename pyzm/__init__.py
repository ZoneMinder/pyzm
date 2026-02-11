"""pyzm -- ZoneMinder Python library.

v2: Typed configuration, clean APIs, proper result objects.
"""

__version__ = "2.0.0"
VERSION = __version__

from pyzm.client import ZMClient
from pyzm.ml.detector import Detector
from pyzm.models.config import (
    DetectorConfig,
    ModelConfig,
    StreamConfig,
    ZMClientConfig,
)
from pyzm.models.detection import BBox, Detection, DetectionResult

__all__ = [
    "ZMClient",
    "Detector",
    "ZMClientConfig",
    "DetectorConfig",
    "ModelConfig",
    "StreamConfig",
    "BBox",
    "Detection",
    "DetectionResult",
]
