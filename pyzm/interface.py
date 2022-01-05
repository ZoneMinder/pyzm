from dataclasses import dataclass, field
import time
from typing import Optional, Union


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass()
class GlobalConfig(metaclass=Singleton):
    """dataclass that holds some global objects"""
    eid: Optional[Union[str, int]] = None  # global Event ID or the name of the input file/video

    DEFAULT_CONFIG: dict = field(default_factory=dict)
    animation_seconds: Optional[time.perf_counter] = None
    logger = None  # global logger, starts with a buffer that is displayed once ZMLog is initialized
    api = None  # global ZMApi
    config: dict = field(default_factory=dict)  # object that will hold active config values from objectconfig.yml
    mid: Optional[int] = None  # global Monitor ID

    api_event_response: Optional[dict] = None  # return from requesting event data
    event_tot_frames: Optional[int] = None  # Total frame buffer length for current event / video /image
    Frame: Optional[list] = None  # Hold the events 'Frame' data structure (length = length of frame buffer)
    Monitor: Optional[dict] = None  # Hold the events 'Monitor' data structure
    Event: Optional[dict] = None  # Hold the events 'Event' data structure

    extras: dict = field(default_factory=dict)  # Extras
