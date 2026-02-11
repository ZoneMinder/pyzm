"""pyzm.zm -- ZoneMinder API layer (v2).

Re-exports the public surface so callers can do::

    from pyzm.zm import ZMAPI, AuthManager, SharedMemory, FrameExtractor
"""

from pyzm.zm.auth import AuthManager
from pyzm.zm.api import ZMAPI
from pyzm.zm.media import FrameExtractor
from pyzm.zm.shm import SharedMemory

__all__ = [
    "AuthManager",
    "ZMAPI",
    "FrameExtractor",
    "SharedMemory",
]
