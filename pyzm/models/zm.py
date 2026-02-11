"""ZoneMinder data models.

Typed representations of ZM resources (monitors, events, frames, zones).
These are plain data objects - the HTTP interaction lives in :mod:`pyzm.zm.api`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Zone:
    """A detection zone polygon belonging to a monitor."""
    name: str
    points: list[tuple[int, int]]
    pattern: str | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.points, "pattern": self.pattern}


@dataclass
class Frame:
    """Metadata for a single event frame."""
    frame_id: int | str
    event_id: int
    type: str = ""        # "Alarm", "Bulk", "Normal", etc.
    score: int = 0
    delta: float = 0.0


@dataclass
class Event:
    """A ZoneMinder event."""
    id: int
    name: str = ""
    monitor_id: int = 0
    cause: str = ""
    notes: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    length: float = 0.0
    frames: int = 0
    alarm_frames: int = 0
    max_score: int = 0
    max_score_frame_id: int | None = None
    storage_path: str = ""

    # Populated lazily by ZMClient
    _frame_list: list[Frame] = field(default_factory=list, repr=False)

    @classmethod
    def from_api_dict(cls, data: dict) -> "Event":
        """Build from a ZM API ``Event`` JSON dict."""
        ev = data.get("Event", data)
        return cls(
            id=int(ev.get("Id", 0)),
            name=ev.get("Name", ""),
            monitor_id=int(ev.get("MonitorId", 0)),
            cause=ev.get("Cause", ""),
            notes=ev.get("Notes", ""),
            start_time=_parse_dt(ev.get("StartTime")),
            end_time=_parse_dt(ev.get("EndTime")),
            length=float(ev.get("Length", 0)),
            frames=int(ev.get("Frames", 0)),
            alarm_frames=int(ev.get("AlarmFrames", 0)),
            max_score=int(ev.get("MaxScore", 0)),
            max_score_frame_id=int(ev["MaxScoreFrameId"]) if ev.get("MaxScoreFrameId") else None,
            storage_path=ev.get("StoragePath", ""),
        )


@dataclass
class MonitorStatus:
    """Runtime status of a monitor."""
    state: str = ""          # "Idle", "Alarm", etc.
    fps: float = 0.0
    capturing: str = "None"  # "None", "Capturing", etc.


@dataclass
class Monitor:
    """A ZoneMinder monitor."""
    id: int
    name: str = ""
    function: str = ""      # "Monitor", "Modect", "Record", etc.
    enabled: bool = True
    width: int = 0
    height: int = 0
    type: str = ""           # "Local", "Remote", "File", "Ffmpeg", etc.
    zones: list[Zone] = field(default_factory=list)
    status: MonitorStatus = field(default_factory=MonitorStatus)

    @classmethod
    def from_api_dict(cls, data: dict) -> "Monitor":
        """Build from a ZM API ``Monitor`` JSON dict."""
        mon = data.get("Monitor", data)
        status_data = data.get("Monitor_Status", {})
        return cls(
            id=int(mon.get("Id", 0)),
            name=mon.get("Name", ""),
            function=mon.get("Function", ""),
            enabled=mon.get("Enabled") == "1",
            width=int(mon.get("Width", 0)),
            height=int(mon.get("Height", 0)),
            type=mon.get("Type", ""),
            status=MonitorStatus(
                state=status_data.get("Status", ""),
                fps=float(status_data.get("CaptureFPS", 0) or 0),
                capturing=status_data.get("Capturing", "None"),
            ),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(val, fmt)
        except (ValueError, TypeError):
            continue
    return None
