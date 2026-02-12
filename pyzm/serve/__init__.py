"""pyzm.serve -- Remote ML detection server.

Run with::

    python -m pyzm.serve --models yolo26s --port 5000
"""

from pyzm.serve.app import create_app

__all__ = ["create_app"]
