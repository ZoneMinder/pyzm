#!/usr/bin/env python3
"""pyzm v2 -- detect objects via a remote pyzm.serve server.

Prerequisites:
    Start the server on the GPU box:
        python -m pyzm.serve --models yolov4 --port 5000

    With authentication:
        python -m pyzm.serve --models yolov4 --port 5000 \
            --auth --auth-user admin --auth-password secret

Usage:
    python remote.py <image_path> [gateway_url]

Examples:
    python remote.py /path/to/image.jpg
    python remote.py /path/to/image.jpg http://gpu-box:5000
"""

import sys

from pyzm import Detector

gateway = "http://localhost:5000"

if len(sys.argv) < 2:
    image_path = input("Enter filename to analyze: ")
else:
    image_path = sys.argv[1]

if len(sys.argv) >= 3:
    gateway = sys.argv[2]

# Remote detection -- image is JPEG-encoded and sent to the server.
# Models are loaded once on the server and persist across requests.
detector = Detector(models=["yolov4"], gateway=gateway)

# For authenticated servers:
# detector = Detector(
#     models=["yolov4"],
#     gateway=gateway,
#     gateway_username="admin",
#     gateway_password="secret",
# )

result = detector.detect(image_path)

print(f"LABELS: {result.labels}")
print(f"BOXES: {result.boxes}")
print(f"CONFIDENCES: {result.confidences}")
print(f"SUMMARY: {result.summary}")
