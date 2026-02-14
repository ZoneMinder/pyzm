"""E2E tests for ZMClient.get_event_frames()."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.zm_e2e


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cv2(), reason="cv2 not installed")
class TestGetEventFrames:
    def test_returns_frames_and_dims(self, zm_client, any_event):
        """get_event_frames should return (frames_list, image_dims_dict)."""
        frames, dims = zm_client.get_event_frames(any_event.id)
        assert isinstance(frames, list)
        assert isinstance(dims, dict)

    def test_frames_not_empty(self, zm_client, any_event):
        """At least one frame should be extracted from a real event."""
        frames, _ = zm_client.get_event_frames(any_event.id)
        assert len(frames) > 0, "Expected at least one frame"

    def test_frame_tuple_structure(self, zm_client, any_event):
        """Each frame should be (frame_id, numpy_array)."""
        import numpy as np

        frames, _ = zm_client.get_event_frames(any_event.id)
        if not frames:
            pytest.skip("No frames extracted")
        frame_id, image = frames[0]
        assert isinstance(frame_id, (int, str))
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3  # (H, W, C)

    def test_image_dims_structure(self, zm_client, any_event):
        """image_dimensions dict should have 'original' and 'resized' keys."""
        _, dims = zm_client.get_event_frames(any_event.id)
        assert "original" in dims
        assert "resized" in dims

    def test_original_dims_are_positive(self, zm_client, any_event):
        """Original dimensions should be a (height, width) tuple of positive ints."""
        _, dims = zm_client.get_event_frames(any_event.id)
        orig = dims["original"]
        if orig is None:
            pytest.skip("No original dimensions reported")
        assert len(orig) == 2
        assert orig[0] > 0 and orig[1] > 0
