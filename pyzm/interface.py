from dataclasses import dataclass, field
import time
from typing import Optional, Union


@dataclass()
class GlobalConfig:
    """dataclass that holds some global objects"""
    from pyzm.ZMLog import ZMLog
    from pyzm.api import ZMApi
    eid: Optional[Union[str, int]] = None  # global Event ID or the name of the input file/video

    DEFAULT_CONFIG: dict = field(default_factory=dict)
    animation_seconds: Optional[time.perf_counter] = None
    logger: Optional[ZMLog] = None  # global logger, starts with a buffer that is displayed once ZMLog is initialized
    api: Optional[ZMApi] = None  # global ZMApi
    config: dict = field(default_factory=dict)  # object that will hold active config values from objectconfig.yml
    mid: Optional[int] = None  # global Monitor ID

    api_event_response: Optional[dict] = None  # return from requesting event data
    event_tot_frames: Optional[int] = None  # Total frame buffer length for current event / video /image
    Frame: Optional[list] = None  # Hold the events 'Frame' data structure (length = length of frame buffer)
    Monitor: Optional[dict] = None  # Hold the events 'Monitor' data structure
    Event: Optional[dict] = None  # Hold the events 'Event' data structure

    extras: dict = field(default_factory=dict)  # Extras
