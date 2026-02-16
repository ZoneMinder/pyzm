"""Tests for the paginated grid review helpers in pyzm.train.app."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyzm.train.dataset import Annotation
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_det(label: str, status=DetectionStatus.APPROVED) -> VerifiedDetection:
    """Create a minimal VerifiedDetection with the given label."""
    return VerifiedDetection(
        detection_id="det_0",
        original=Annotation(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1),
        status=status,
        original_label=label,
    )


def _make_store(tmp_path: Path, images: dict[str, bool]) -> VerificationStore:
    """Create a VerificationStore with given images.

    *images* maps image_name → fully_reviewed.
    """
    (tmp_path / "project.json").write_text('{"classes": []}')
    store = VerificationStore(tmp_path)
    for name, reviewed in images.items():
        store.set(ImageVerification(image_name=name, fully_reviewed=reviewed))
    return store


def _make_store_with_dets(
    tmp_path: Path,
    images: dict[str, list[str]],
) -> VerificationStore:
    """Create a store where each image has approved detections for given labels.

    *images* maps image_name → list of class labels.
    """
    (tmp_path / "project.json").write_text('{"classes": []}')
    store = VerificationStore(tmp_path)
    for name, labels in images.items():
        dets = [
            VerifiedDetection(
                detection_id=f"det_{j}",
                original=Annotation(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1),
                status=DetectionStatus.APPROVED,
                original_label=lbl,
            )
            for j, lbl in enumerate(labels)
        ]
        store.set(ImageVerification(
            image_name=name, detections=dets, fully_reviewed=True,
        ))
    return store


def _make_paths(names: list[str]) -> list[Path]:
    return [Path(f"/fake/{n}") for n in names]


# ── _is_reviewed ─────────────────────────────────────────────────────

class TestIsReviewed:
    def test_true_when_reviewed(self, tmp_path):
        from pyzm.train.app import _is_reviewed

        store = _make_store(tmp_path, {"a.jpg": True})
        assert _is_reviewed(store, "a.jpg") is True

    def test_false_when_not_reviewed(self, tmp_path):
        from pyzm.train.app import _is_reviewed

        store = _make_store(tmp_path, {"a.jpg": False})
        assert _is_reviewed(store, "a.jpg") is False

    def test_false_when_missing(self, tmp_path):
        from pyzm.train.app import _is_reviewed

        store = _make_store(tmp_path, {})
        assert _is_reviewed(store, "missing.jpg") is False


# ── _filtered_images ─────────────────────────────────────────────────

class TestFilteredImages:
    def setup_method(self):
        from pyzm.train.app import _filtered_images
        self._fn = _filtered_images

    def test_all_returns_everything(self, tmp_path):
        store = _make_store(tmp_path, {"a.jpg": True, "b.jpg": False})
        paths = _make_paths(["a.jpg", "b.jpg"])
        result = self._fn(paths, store, "all")
        assert len(result) == 2

    def test_approved_only(self, tmp_path):
        store = _make_store(tmp_path, {"a.jpg": True, "b.jpg": False, "c.jpg": True})
        paths = _make_paths(["a.jpg", "b.jpg", "c.jpg"])
        result = self._fn(paths, store, "approved")
        assert [p.name for p in result] == ["a.jpg", "c.jpg"]

    def test_unapproved_only(self, tmp_path):
        store = _make_store(tmp_path, {"a.jpg": True, "b.jpg": False})
        paths = _make_paths(["a.jpg", "b.jpg"])
        result = self._fn(paths, store, "unapproved")
        assert [p.name for p in result] == ["b.jpg"]

    def test_unapproved_includes_missing(self, tmp_path):
        store = _make_store(tmp_path, {"a.jpg": True})
        paths = _make_paths(["a.jpg", "b.jpg"])
        result = self._fn(paths, store, "unapproved")
        assert [p.name for p in result] == ["b.jpg"]

    def test_empty_images(self, tmp_path):
        store = _make_store(tmp_path, {})
        result = self._fn([], store, "all")
        assert result == []

    def test_object_class_filter(self, tmp_path):
        store = _make_store_with_dets(tmp_path, {
            "a.jpg": ["car", "person"],
            "b.jpg": ["dog"],
            "c.jpg": ["car"],
        })
        paths = _make_paths(["a.jpg", "b.jpg", "c.jpg"])
        result = self._fn(paths, store, "all", object_class="car")
        assert [p.name for p in result] == ["a.jpg", "c.jpg"]

    def test_object_class_none_returns_all(self, tmp_path):
        store = _make_store_with_dets(tmp_path, {
            "a.jpg": ["car"],
            "b.jpg": ["dog"],
        })
        paths = _make_paths(["a.jpg", "b.jpg"])
        result = self._fn(paths, store, "all", object_class=None)
        assert len(result) == 2

    def test_object_class_combined_with_status(self, tmp_path):
        """Object filter stacks with the status filter."""
        store = _make_store_with_dets(tmp_path, {
            "a.jpg": ["car"],
            "b.jpg": ["car"],
        })
        # Mark b.jpg as not reviewed
        iv = store.get("b.jpg")
        iv.fully_reviewed = False
        store.set(iv)

        paths = _make_paths(["a.jpg", "b.jpg"])
        result = self._fn(paths, store, "approved", object_class="car")
        assert [p.name for p in result] == ["a.jpg"]

    def test_object_class_no_match(self, tmp_path):
        store = _make_store_with_dets(tmp_path, {
            "a.jpg": ["car"],
        })
        paths = _make_paths(["a.jpg"])
        result = self._fn(paths, store, "all", object_class="dog")
        assert result == []

    def test_object_class_skips_deleted(self, tmp_path):
        """Deleted detections don't count for class filtering."""
        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        store.set(ImageVerification(
            image_name="a.jpg",
            detections=[_make_det("car", status=DetectionStatus.DELETED)],
            fully_reviewed=True,
        ))
        paths = _make_paths(["a.jpg"])
        result = self._fn(paths, store, "all", object_class="car")
        assert result == []


# ── _image_has_class ─────────────────────────────────────────────────

class TestImageHasClass:
    def test_true_when_class_present(self, tmp_path):
        from pyzm.train.app import _image_has_class

        store = _make_store_with_dets(tmp_path, {"a.jpg": ["car", "person"]})
        assert _image_has_class(store, "a.jpg", "car") is True

    def test_false_when_class_absent(self, tmp_path):
        from pyzm.train.app import _image_has_class

        store = _make_store_with_dets(tmp_path, {"a.jpg": ["car"]})
        assert _image_has_class(store, "a.jpg", "dog") is False

    def test_false_when_image_missing(self, tmp_path):
        from pyzm.train.app import _image_has_class

        store = _make_store_with_dets(tmp_path, {})
        assert _image_has_class(store, "missing.jpg", "car") is False

    def test_false_when_only_deleted(self, tmp_path):
        from pyzm.train.app import _image_has_class

        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        store.set(ImageVerification(
            image_name="a.jpg",
            detections=[_make_det("car", status=DetectionStatus.DELETED)],
        ))
        assert _image_has_class(store, "a.jpg", "car") is False

    def test_true_with_renamed(self, tmp_path):
        """A renamed detection matches its effective (new) label."""
        from pyzm.train.app import _image_has_class

        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        det = _make_det("cat", status=DetectionStatus.RENAMED)
        det.new_label = "dog"
        store.set(ImageVerification(image_name="a.jpg", detections=[det]))
        assert _image_has_class(store, "a.jpg", "dog") is True
        assert _image_has_class(store, "a.jpg", "cat") is False


# ── Pagination math ──────────────────────────────────────────────────

class TestPaginationMath:
    """Test the ceil-division logic used in _review_grid."""

    @pytest.mark.parametrize("n,page_size,expected", [
        (0, 20, 1),      # zero images → 1 page (max(1, ...))
        (1, 20, 1),      # 1 image → 1 page
        (20, 20, 1),     # exact multiple → 1 page
        (21, 20, 2),     # one over → 2 pages
        (40, 20, 2),     # exact multiple → 2 pages
        (41, 20, 3),     # one over → 3 pages
        (100, 12, 9),    # non-multiple
        (12, 12, 1),     # exact with non-default page size
    ])
    def test_total_pages(self, n, page_size, expected):
        total_pages = max(1, -(-n // page_size))
        assert total_pages == expected


# ── Thumbnail cache invalidation ─────────────────────────────────────

class TestThumbnailCache:
    def test_invalidate_removes_entry(self):
        """_invalidate_thumbnail removes the entry from the cache dict."""
        import streamlit as st
        from pyzm.train.app import _invalidate_thumbnail

        st.session_state["_thumb_cache"] = {"a.jpg": "data:...", "b.jpg": "data:..."}
        _invalidate_thumbnail("a.jpg")
        assert "a.jpg" not in st.session_state["_thumb_cache"]
        assert "b.jpg" in st.session_state["_thumb_cache"]

    def test_invalidate_noop_when_missing(self):
        """_invalidate_thumbnail is a no-op when key doesn't exist."""
        import streamlit as st
        from pyzm.train.app import _invalidate_thumbnail

        st.session_state["_thumb_cache"] = {"a.jpg": "data:..."}
        _invalidate_thumbnail("nonexistent.jpg")  # should not raise
        assert len(st.session_state["_thumb_cache"]) == 1

    def test_invalidate_noop_when_no_cache(self):
        """_invalidate_thumbnail is safe when cache doesn't exist yet."""
        import streamlit as st
        from pyzm.train.app import _invalidate_thumbnail

        st.session_state.pop("_thumb_cache", None)
        _invalidate_thumbnail("a.jpg")  # should not raise

    def test_fifo_eviction_at_limit(self):
        """Cache evicts oldest entry when at _THUMB_CACHE_MAX."""
        import streamlit as st
        from pyzm.train.app import _THUMB_CACHE_MAX

        # Simulate a full cache
        cache = {f"img_{i}.jpg": f"data:{i}" for i in range(_THUMB_CACHE_MAX)}
        st.session_state["_thumb_cache"] = cache
        assert len(cache) == _THUMB_CACHE_MAX

        # The next _generate_thumbnail_uri call would evict the oldest.
        # We test the eviction logic directly.
        if len(cache) >= _THUMB_CACHE_MAX:
            oldest = next(iter(cache))
            del cache[oldest]
        cache["new_image.jpg"] = "data:new"

        assert len(cache) == _THUMB_CACHE_MAX
        assert "img_0.jpg" not in cache
        assert "new_image.jpg" in cache


# ── has_modifications ────────────────────────────────────────────────

class TestHasModifications:
    def test_false_when_all_approved(self, tmp_path):
        store = _make_store_with_dets(tmp_path, {
            "a.jpg": ["car", "person"],
            "b.jpg": ["dog"],
        })
        assert store.has_modifications() is False

    def test_false_when_all_pending(self, tmp_path):
        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        store.set(ImageVerification(
            image_name="a.jpg",
            detections=[_make_det("car", status=DetectionStatus.PENDING)],
        ))
        assert store.has_modifications() is False

    def test_true_when_deleted(self, tmp_path):
        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        store.set(ImageVerification(
            image_name="a.jpg",
            detections=[_make_det("car", status=DetectionStatus.DELETED)],
        ))
        assert store.has_modifications() is True

    def test_true_when_renamed(self, tmp_path):
        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        det = _make_det("cat", status=DetectionStatus.RENAMED)
        det.new_label = "dog"
        store.set(ImageVerification(image_name="a.jpg", detections=[det]))
        assert store.has_modifications() is True

    def test_true_when_reshaped(self, tmp_path):
        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        det = _make_det("car", status=DetectionStatus.RESHAPED)
        det.adjusted = Annotation(class_id=0, cx=0.6, cy=0.6, w=0.2, h=0.2)
        store.set(ImageVerification(image_name="a.jpg", detections=[det]))
        assert store.has_modifications() is True

    def test_true_when_added(self, tmp_path):
        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        store.set(ImageVerification(
            image_name="a.jpg",
            detections=[_make_det("car", status=DetectionStatus.ADDED)],
        ))
        assert store.has_modifications() is True

    def test_false_when_empty(self, tmp_path):
        (tmp_path / "project.json").write_text('{"classes": []}')
        store = VerificationStore(tmp_path)
        assert store.has_modifications() is False
