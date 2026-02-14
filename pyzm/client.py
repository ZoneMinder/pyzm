"""Top-level ZoneMinder client -- the one-stop public API.

Usage::

    from pyzm import ZMClient

    zm = ZMClient(url="https://zm.example.com/zm/api", user="admin", password="secret")

    for m in zm.monitors():
        print(m.name, m.function)

    for ev in zm.events(monitor_id=1, since="1 hour ago"):
        print(ev.id, ev.cause)

    frames, dims = zm.get_event_frames(12345)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pyzm.models.config import StreamConfig, ZMClientConfig
from pyzm.models.zm import Event, Monitor, Zone
from pyzm.zm.api import ZMAPI
from pyzm.zm.media import FrameExtractor

logger = logging.getLogger("pyzm")


class ZMClient:
    """High-level ZoneMinder client.

    Parameters
    ----------
    url:
        ZM URL -- either the API URL (``https://server/zm/api``) or the
        portal URL (``https://server/zm``).  If ``/api`` is missing it is
        appended automatically.
    user:
        ZM username.  ``None`` when auth is disabled.
    password:
        ZM password.
    portal_url:
        Full portal URL (e.g. ``https://server/zm``).  Auto-derived from
        *url* when not provided.
    verify_ssl:
        Whether to verify SSL certificates.  Set to ``False`` for
        self-signed certs.
    config:
        A pre-built :class:`ZMClientConfig`.  When provided, all other
        keyword args are ignored.
    """

    def __init__(
        self,
        url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        portal_url: str | None = None,
        verify_ssl: bool = True,
        timeout: int = 30,
        config: ZMClientConfig | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            if url is None:
                raise ValueError("Either 'url' or 'config' must be provided")
            # Auto-append /api if the user gave us the portal URL
            api_url = url.rstrip("/")
            if not api_url.endswith("/api"):
                logger.debug("URL %r does not end with /api, appending it", url)
                api_url = api_url + "/api"
            self._config = ZMClientConfig(
                api_url=api_url,
                portal_url=portal_url,
                user=user,
                password=password,
                verify_ssl=verify_ssl,
                timeout=timeout,
            )

        self._api = ZMAPI(self._config)

        # Caches
        self._monitors: list[Monitor] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def api(self) -> ZMAPI:
        """The underlying low-level API (for advanced use)."""
        return self._api

    @property
    def zm_version(self) -> str | None:
        return self._api.zm_version

    @property
    def api_version(self) -> str | None:
        return self._api.api_version

    # ------------------------------------------------------------------
    # Monitors
    # ------------------------------------------------------------------

    def monitors(self, *, force_reload: bool = False) -> list[Monitor]:
        """Return all monitors.  Cached after first call."""
        if self._monitors is not None and not force_reload:
            return self._monitors

        data = self._api.get("monitors.json")
        raw_list = data.get("monitors", []) if data else []
        self._monitors = [Monitor.from_api_dict(m) for m in raw_list]
        return self._monitors

    def monitor(self, monitor_id: int) -> Monitor:
        """Return a single monitor by ID."""
        for m in self.monitors():
            if m.id == monitor_id:
                return m
        # Fallback: direct API call
        data = self._api.get(f"monitors/{monitor_id}.json")
        if data and data.get("monitor"):
            return Monitor.from_api_dict(data["monitor"])
        raise ValueError(f"Monitor {monitor_id} not found")

    def monitor_zones(self, monitor_id: int) -> list[Zone]:
        """Return detection zones for a monitor."""
        data = self._api.get(f"zones/forMonitor/{monitor_id}.json")
        zones: list[Zone] = []
        for z in data.get("zones", []) if data else []:
            zd = z.get("Zone", z)
            coords_str = zd.get("Coords", "")
            points = _parse_zone_coords(coords_str)
            zones.append(Zone(
                name=zd.get("Name", ""),
                points=points,
            ))
        return zones

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def events(
        self,
        *,
        event_id: int | None = None,
        monitor_id: int | None = None,
        since: str | None = None,
        until: str | None = None,
        min_alarm_frames: int | None = None,
        object_only: bool = False,
        limit: int = 100,
    ) -> list[Event]:
        """Query events with optional filters."""
        # Use CakePHP URL-path filter syntax (e.g. events/index/Field:val)
        # instead of filter[Query][terms] query strings which are broken
        # on ZM >= 1.38.  REGEXP replaces LIKE to avoid % in URL paths.
        path_parts: list[str] = []

        if event_id is not None:
            path_parts.append(f"Id:{event_id}")

        if monitor_id is not None:
            path_parts.append(f"MonitorId:{monitor_id}")

        if since:
            parsed = _parse_human_time(since)
            if parsed:
                path_parts.append(f"StartTime >=:{parsed}")

        if until:
            parsed = _parse_human_time(until)
            if parsed:
                path_parts.append(f"StartTime <=:{parsed}")

        if min_alarm_frames is not None:
            path_parts.append(f"AlarmFrames >=:{min_alarm_frames}")

        if object_only:
            path_parts.append("Notes REGEXP:detected")

        path_filter = "/".join(path_parts)
        endpoint = f"events/index/{path_filter}.json" if path_filter else "events/index.json"
        params = {"page": "1", "limit": str(limit)}

        data = self._api.get(endpoint, params=params)
        events_list = data.get("events", []) if data else []
        return [Event.from_api_dict(e) for e in events_list]

    def event(self, event_id: int) -> Event:
        """Fetch a single event by ID."""
        data = self._api.get(f"events/{event_id}.json")
        if data and data.get("event"):
            return Event.from_api_dict(data["event"])
        raise ValueError(f"Event {event_id} not found")

    def update_event_notes(self, event_id: int, notes: str) -> None:
        """Update the Notes field of an event."""
        url = f"events/{event_id}.json"
        self._api.put(url, data={"Event[Notes]": notes})

    def tag_event(self, event_id: int, labels: list[str]) -> None:
        """Tag an event with detected object labels.

        For each unique label, creates the tag if it doesn't exist and
        associates it with the event.  Requires ZM >= 1.37.44.

        Parameters
        ----------
        event_id:
            ZoneMinder event ID.
        labels:
            Detection labels to tag (e.g. ``["person", "car"]``).
            Duplicates are ignored.
        """
        unique = list(dict.fromkeys(labels))  # dedupe, preserve order
        if not unique:
            return
        logger.debug("Tagging event %s with %s", event_id, unique)
        for label in unique:
            self._tag_one(label, event_id)

    def _tag_one(self, label: str, event_id: int) -> None:
        """Create or find a tag by name and link it to an event via direct DB."""
        import datetime
        try:
            from pyzm.zm.db import get_zm_db
        except ImportError:
            logger.warning("pyzm.zm.db not available, cannot tag events")
            return

        conn = get_zm_db()
        if conn is None:
            logger.warning("Could not connect to ZM database, skipping tagging")
            return

        cur = conn.cursor(dictionary=True)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Find or create tag
        cur.execute("SELECT Id FROM Tags WHERE Name=%s", (label,))
        row = cur.fetchone()
        if row:
            tag_id = row["Id"]
            cur.execute("UPDATE Tags SET LastAssignedDate=%s WHERE Id=%s", (now, tag_id))
            logger.debug("Tag '%s' exists (id=%s), linked to event %s", label, tag_id, event_id)
        else:
            cur.execute(
                "INSERT INTO Tags (Name, CreateDate, LastAssignedDate) VALUES (%s, %s, %s)",
                (label, now, now),
            )
            tag_id = cur.lastrowid
            logger.debug("Created tag '%s' (id=%s), linked to event %s", label, tag_id, event_id)

        # Link tag to event (ignore duplicate)
        try:
            cur.execute(
                "INSERT INTO Events_Tags (TagId, EventId) VALUES (%s, %s)",
                (tag_id, event_id),
            )
        except Exception:
            logger.debug("Tag '%s' already linked to event %s", label, event_id)

        conn.commit()
        cur.close()

    # ------------------------------------------------------------------
    # Event path
    # ------------------------------------------------------------------

    def event_path(self, event_id: int) -> str | None:
        """Construct the filesystem path for an event, same as ZoneMinder::Event->Path().

        Queries the DB for StoragePath, Scheme, MonitorId, and StartDateTime,
        then builds the relative path based on the storage scheme.
        """
        from datetime import datetime as _dt
        from pyzm.zm.db import get_zm_db

        conn = get_zm_db()
        if conn is None:
            logger.warning("Cannot resolve event path: DB unavailable")
            return None

        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT E.MonitorId, E.StartDateTime, S.Path AS StoragePath, S.Scheme "
            "FROM Events E JOIN Storage S ON E.StorageId = S.Id WHERE E.Id=%s",
            (event_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row or not row["StoragePath"] or not row["StartDateTime"]:
            logger.warning("Cannot resolve event path: missing DB fields for event %s", event_id)
            return None

        storage_path = row["StoragePath"]
        monitor_id = row["MonitorId"]
        scheme = (row.get("Scheme") or "Medium").capitalize()
        start_dt = row["StartDateTime"]
        if isinstance(start_dt, str):
            start_dt = _dt.strptime(start_dt, "%Y-%m-%d %H:%M:%S")

        if scheme == "Deep":
            relative = "{}/{}".format(monitor_id, start_dt.strftime("%y/%m/%d/%H/%M/%S"))
        elif scheme == "Medium":
            relative = "{}/{}/{}".format(monitor_id, start_dt.strftime("%Y-%m-%d"), event_id)
        else:
            relative = "{}/{}".format(monitor_id, event_id)

        path = os.path.join(storage_path, relative)
        logger.debug("Event %s path (scheme=%s): %s", event_id, scheme, path)
        return path

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def get_event_frames(
        self,
        event_id: int,
        stream_config: StreamConfig | None = None,
    ) -> tuple[list[tuple[int | str, Any]], dict[str, tuple[int, int] | None]]:
        """Extract frames from a ZM event.

        Returns
        -------
        tuple[list[tuple[frame_id, ndarray]], dict]
            A pair of ``(frames, image_dimensions)`` where *frames* is a
            list of ``(frame_id, numpy_array)`` tuples and
            *image_dimensions* is ``{'original': (h,w), 'resized': (rh,rw)|None}``.
        """
        sc = stream_config or StreamConfig()
        extractor = FrameExtractor(api=self._api, stream_config=sc)
        frames: list[tuple[int | str, Any]] = []
        for frame, img in extractor.extract_frames(str(event_id)):
            frames.append((frame.frame_id, img))

        orig = extractor.original_shape
        if frames and orig:
            resized_h, resized_w = frames[0][1].shape[:2]
            resized = (resized_h, resized_w) if (resized_h, resized_w) != orig else None
        else:
            resized = None

        image_dims: dict[str, tuple[int, int] | None] = {
            "original": orig,
            "resized": resized,
        }
        return frames, image_dims

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state(self, state: str) -> Any:
        """Set ZM to a named state (e.g. 'start', 'stop', 'restart')."""
        return self._api.get(f"states/change/{state}.json")

    def start(self) -> Any:
        return self.set_state("start")

    def stop(self) -> Any:
        return self.set_state("stop")

    def restart(self) -> Any:
        return self.set_state("restart")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_zone_coords(coords_str: str) -> list[tuple[int, int]]:
    """Parse ZM zone coordinate string like ``"0,0 639,0 639,479 0,479"``."""
    if not coords_str:
        return []
    points: list[tuple[int, int]] = []
    for pair in coords_str.strip().split():
        parts = pair.split(",")
        if len(parts) == 2:
            points.append((int(parts[0]), int(parts[1])))
    return points


def _parse_human_time(time_str: str) -> str | None:
    """Parse human-readable time strings into ISO format.

    Supports formats like ``"1 hour ago"``, ``"2024-01-15 10:30:00"``.
    Falls back to returning the string as-is for ZM to parse.
    """
    try:
        import dateparser
        dt = dateparser.parse(time_str)
        if dt:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ImportError:
        pass
    return time_str
