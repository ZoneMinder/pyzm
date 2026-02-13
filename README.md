<img src="https://raw.githubusercontent.com/ZoneMinder/pyzm/master/images/pyzm.png" width="200"/>

What
=====
Pythonic ZoneMinder wrapper
- API
- Event Server
- Logger
- Memory
- Machine Learning Modules

Installation
=============
See the [installation guide](https://pyzmv2.readthedocs.io/en/latest/guide/installation.html) on ReadTheDocs.

Documentation & Examples
=========================
Latest documentation is available <a href='https://pyzmv2.readthedocs.io/en/latest/'>here</a>. The documentation includes a full example.

Features
=========
- API auth using tokens or legacy (manages refresh logins automatically)
- Monitors
- Events with filters
- States
- Configs
- EventNotification callbacks
- Mapped Memory access
- Direct access to ML algorithms
- Remote ML detection server (`pyzm.serve`) — run models on a GPU box, detect from anywhere
- [Amazon Rekognition support](https://medium.com/@michael-ludvig/aws-rekognition-support-for-zoneminder-object-detection-40b71f926a80) for object detection

Testing
========

pyzm has two test tiers:

**Unit / integration tests** (no hardware required):
```bash
pip install pytest
python -m pytest tests/ -m "not e2e" -v
```

**End-to-end tests** (require real ML models on disk):
```bash
# Requires models in /var/lib/zmeventnotification/models/
# and the test image at tests/test_e2e/bird.jpg (included in repo)
python -m pytest tests/test_e2e/ -v

# Skip the slower remote-serve tests:
python -m pytest tests/test_e2e/ -v -m "not serve"

# Run only remote-serve tests:
python -m pytest tests/test_e2e/ -v -m serve
```

The e2e suite covers every objectconfig feature: pattern matching, zone/polygon filtering,
size filtering, min_confidence, disabled models, match strategies, frame strategies,
pre_existing_labels, match_past_detections (aliases, ignore_labels, per-label overrides),
Detector.from_dict, StreamConfig.from_dict, lazy/eager pipeline loading, remote pyzm.serve
(health, detect, /models, --models all, auth, gateway mode), and more.

Limitations
============
* Only for Python3



