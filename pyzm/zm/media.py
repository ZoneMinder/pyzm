"""Frame extraction from ZoneMinder events and media files.

Supports three source types:

* **Image file** -- a local ``.jpg`` / ``.png`` path.
* **Video file** -- a local video path (read via OpenCV).
* **ZM API** -- numeric event ID fetched frame-by-frame through
  ``index.php`` (the ZM image-serving endpoint).

Heavy dependencies (``cv2``, ``numpy``) are imported at function level so
that this module can be imported even when they are not installed.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from pyzm.models.config import StreamConfig
from pyzm.models.zm import Frame

if TYPE_CHECKING:
    import numpy as np
    from pyzm.zm.api import ZMAPI

logger = logging.getLogger("pyzm.zm")


class FrameExtractor:
    """Extract image frames from an event or media file.

    Parameters
    ----------
    api:
        A connected :class:`~pyzm.zm.api.ZMAPI` instance.  May be
        ``None`` when extracting from local files only.
    stream_config:
        Controls frame selection, resizing, retries, etc.
    """

    def __init__(
        self,
        api: ZMAPI | None,
        stream_config: StreamConfig,
    ) -> None:
        self._api = api
        self._cfg = stream_config
        self._original_shape: tuple[int, int] | None = None

    @property
    def original_shape(self) -> tuple[int, int] | None:
        """``(height, width)`` of the image before resizing, or ``None``."""
        return self._original_shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_frames(
        self, stream: str,
    ) -> Generator[tuple[Frame, Any], None, None]:
        """Yield ``(Frame, image_array)`` tuples from *stream*.

        *stream* can be:

        * A path to a local image file (``.jpg``, ``.png``, ``.jpeg``).
        * A path to a local video file.
        * A numeric event ID (as a string) -- frames are fetched via the
          ZM ``index.php`` image endpoint.

        Yields
        ------
        tuple[Frame, numpy.ndarray]
            Frame metadata and the decoded image (BGR, uint8).
        """
        import numpy as np  # noqa: F811

        stream = stream.strip()

        if self._cfg.delay:
            logger.debug("Initial delay: %d s", self._cfg.delay)
            time.sleep(self._cfg.delay)

        _, ext = os.path.splitext(stream)
        if ext.lower() in (".jpg", ".jpeg", ".png"):
            yield from self._read_image_file(stream)
        elif stream.isnumeric():
            yield from self._read_zm_event(stream)
        else:
            yield from self._read_video_file(stream)

    # ------------------------------------------------------------------
    # Image file
    # ------------------------------------------------------------------

    def _read_image_file(
        self, path: str,
    ) -> Generator[tuple[Frame, Any], None, None]:
        import cv2

        logger.debug("Reading image file: %s", path)
        img = cv2.imread(path)
        if img is None:
            logger.error("Failed to read image file: %s", path)
            return

        img = self._maybe_resize(img)
        frame = Frame(frame_id=1, event_id=0, type="file")
        yield frame, img

    # ------------------------------------------------------------------
    # Video file
    # ------------------------------------------------------------------

    def _read_video_file(
        self, path: str,
    ) -> Generator[tuple[Frame, Any], None, None]:
        import cv2

        logger.debug("Reading video file: %s", path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error("Failed to open video: %s", path)
            return

        try:
            frame_set = self._resolve_frame_set(is_api=False)
            frame_idx = 0
            frames_yielded = 0
            contiguous_errors = 0

            while cap.isOpened():
                ok, img = cap.read()
                frame_idx += 1

                if not ok or img is None:
                    contiguous_errors += 1
                    if contiguous_errors >= self._cfg.contig_frames_before_error:
                        logger.error("Too many contiguous read errors; stopping")
                        return
                    continue
                contiguous_errors = 0

                if not self._should_process_frame(frame_idx, frame_set, frames_yielded):
                    continue

                img = self._maybe_resize(img)
                self._maybe_save(img, path, frame_idx)

                fid = frame_set[frames_yielded] if frame_set else str(frame_idx)
                frame = Frame(frame_id=fid, event_id=0, type="video")
                yield frame, img

                frames_yielded += 1
                if self._reached_limit(frames_yielded, frame_set):
                    return

                self._inter_frame_delay(frames_yielded)
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # ZM API (index.php image fetching)
    # ------------------------------------------------------------------

    def _read_zm_event(
        self, eid: str,
    ) -> Generator[tuple[Frame, Any], None, None]:
        import cv2
        import numpy as np

        if self._api is None:
            raise ValueError(
                f"Cannot fetch event {eid} via ZM API: no ZMAPI instance provided"
            )

        base_url = (
            f"{self._api.portal_url}/index.php"
            f"?view=image&eid={eid}"
        )
        logger.debug("Streaming event %s from %s", eid, base_url)

        frame_set = self._resolve_frame_set(is_api=True)

        # If frame_set contains "snapshot", try to resolve to the real frame ID
        frame_set = self._resolve_snapshots(frame_set, eid)

        frames_yielded = 0
        next_fid = self._cfg.start_frame
        contiguous_errors = 0

        while True:
            if frame_set:
                if frames_yielded >= len(frame_set):
                    return
                fid = frame_set[frames_yielded]
            else:
                fid = str(next_fid)

            if self._cfg.delay_between_snapshots and fid == "snapshot" and frames_yielded > 0:
                time.sleep(self._cfg.delay_between_snapshots)

            url = f"{base_url}&fid={fid}"
            logger.debug("Fetching frame: %s", url)

            img = self._fetch_frame_image(url, cv2, np)
            if img is None:
                contiguous_errors += 1
                if contiguous_errors >= self._cfg.contig_frames_before_error:
                    logger.error("Too many contiguous fetch errors; stopping")
                    return
                # Advance to next frame in set, or bail
                if frame_set:
                    frames_yielded += 1
                    continue
                else:
                    next_fid += self._cfg.frame_skip
                    continue

            contiguous_errors = 0
            img = self._maybe_resize(img)
            self._maybe_save(img, f"{eid}-image", fid)

            frame = Frame(
                frame_id=fid,
                event_id=int(eid),
                type="Alarm" if fid == "alarm" else "",
            )
            yield frame, img

            frames_yielded += 1
            if self._reached_limit(frames_yielded, frame_set):
                return

            if not frame_set:
                next_fid += self._cfg.frame_skip

            self._inter_frame_delay(frames_yielded)

    def _fetch_frame_image(self, url: str, cv2: Any, np: Any) -> Any:
        """Fetch a single frame image from the ZM API, with retries."""
        for attempt in range(1, self._cfg.max_attempts + 1):
            try:
                resp = self._api.request(url)  # type: ignore[union-attr]
                if resp is None:
                    raise ValueError("BAD_IMAGE")

                # If the API returned a Response object (image content-type)
                if hasattr(resp, "content"):
                    buf = np.asarray(bytearray(resp.content), dtype="uint8")
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                    logger.debug("cv2.imdecode returned None for frame")
                    raise ValueError("BAD_IMAGE")

                # JSON response (shouldn't happen for image fetches)
                logger.debug("Got JSON instead of image for frame URL")
                raise ValueError("BAD_IMAGE")

            except ValueError as exc:
                if str(exc) != "BAD_IMAGE":
                    raise
                logger.debug(
                    "Bad image on attempt %d/%d", attempt, self._cfg.max_attempts,
                )
                if attempt < self._cfg.max_attempts and self._cfg.sleep_between_attempts:
                    time.sleep(self._cfg.sleep_between_attempts)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_frame_set(self, *, is_api: bool) -> list[str]:
        """Return the explicit list of frame IDs/names to process, or []."""
        raw = self._cfg.frame_set
        if not raw:
            return []

        frame_set = list(raw)

        # "alarm" and "snapshot" only work via ZM API
        has_special = any(f in ("alarm", "snapshot") for f in frame_set)
        if has_special and not is_api:
            raise ValueError(
                "Frame types 'alarm'/'snapshot' require ZM API access, "
                "but the stream is a local file"
            )

        return frame_set

    def _resolve_snapshots(self, frame_set: list[str], eid: str) -> list[str]:
        """Replace 'snapshot' entries with the real MaxScoreFrameId if configured."""
        if not self._cfg.convert_snapshot_to_fid or self._api is None:
            return frame_set

        result = list(frame_set)
        for i, fid in enumerate(result):
            if fid != "snapshot":
                continue
            try:
                ev_data = self._api.get(f"events/{eid}.json")
                real_fid = (
                    ev_data.get("event", {})
                    .get("Event", {})
                    .get("MaxScoreFrameId")
                )
                if real_fid:
                    logger.debug(
                        "Resolved snapshot to frame %s for event %s", real_fid, eid,
                    )
                    result[i] = str(real_fid)
            except Exception:
                logger.debug("Failed to resolve snapshot frame ID for event %s", eid)
        return result

    def _should_process_frame(
        self,
        frame_idx: int,
        frame_set: list[str],
        frames_yielded: int,
    ) -> bool:
        """Decide whether *frame_idx* should be yielded."""
        if frame_set:
            # frame_set entries are 1-based frame IDs
            if frames_yielded >= len(frame_set):
                return False
            return frame_idx == int(frame_set[frames_yielded])

        if frame_idx < self._cfg.start_frame:
            return False
        if (frame_idx - self._cfg.start_frame) % self._cfg.frame_skip != 0:
            return False
        return True

    def _reached_limit(self, frames_yielded: int, frame_set: list[str]) -> bool:
        """True when we have yielded enough frames."""
        if frame_set:
            return frames_yielded >= len(frame_set)
        if self._cfg.max_frames > 0:
            return frames_yielded >= self._cfg.max_frames
        return False

    def _maybe_resize(self, img: Any) -> Any:
        """Resize *img* to ``self._cfg.resize`` width if configured."""
        import cv2

        h, w = img.shape[:2]

        if self._cfg.resize is None or w <= self._cfg.resize:
            # Store original shape even when not resizing (first frame wins)
            if self._original_shape is None:
                self._original_shape = (h, w)
            return img

        # Store original dimensions before resize (first frame wins)
        if self._original_shape is None:
            self._original_shape = (h, w)

        ratio = self._cfg.resize / w
        new_w = self._cfg.resize
        new_h = int(h * ratio)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _maybe_save(self, img: Any, basename: str, fid: int | str) -> None:
        """Save the frame to disk if ``save_frames`` is enabled."""
        if not self._cfg.save_frames:
            return

        import cv2

        fname = os.path.join(self._cfg.save_frames_dir, f"{basename}-{fid}.jpg")
        logger.debug("Saving frame to %s", fname)
        cv2.imwrite(fname, img)

    def _inter_frame_delay(self, frames_yielded: int) -> None:
        """Sleep between frames if configured."""
        if self._cfg.delay_between_frames and frames_yielded > 0:
            time.sleep(self._cfg.delay_between_frames)
