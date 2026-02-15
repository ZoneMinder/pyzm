"""Abstract base class for all ML inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from pyzm.models.detection import Detection


class MLBackend(ABC):
    """Abstract base for ML inference backends.

    Every concrete backend must implement :meth:`load`, :meth:`detect`, and
    the :attr:`name` property.  The host pipeline calls ``load()`` once and
    then ``detect()`` per-frame.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def detect(self, image: "np.ndarray") -> list[Detection]:
        """Run inference on a single image, return raw detections."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend instance."""

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def needs_exclusive_lock(self) -> bool:
        """True if this backend requires exclusive hardware (e.g. EdgeTPU)."""
        return False

    def acquire_lock(self) -> None:
        """Acquire the hardware lock. No-op by default."""

    def release_lock(self) -> None:
        """Release the hardware lock. No-op by default."""
