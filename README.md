<img src="https://raw.githubusercontent.com/pliablepixels/pyzm/master/images/pyzm.png" width="200"/>

# *** This is the 'neo-ZMES' forked version! ***

# Note
All credit goes to the original author @pliablepixels. Please see https://github.com/pliablepixels

I taught myself python to work on this project, I am learning git, etc. Please forgive the terrible commits.

Please be aware that the 'neo' versions are NOT compatible with the source repos. The module structure is different,
functions and args are different and processing the configs are a completely different syntax and structure. My goal 
is to add some more options for power users and speed things up. In my personal testing I can say that I have blazing
fast detections compared to the source repos. Gotify is basically instant as long as the app is not battery optimized 
(I am unaware of if gotify has an iOS app).

**I am actively taking enhancement requests for new features and improvements.**

MAJOR CHANGES
---
- PERFORMANCE - I made many, many changes based on becoming more performant. There is logic for live and past events, the frame buffer is smarter and tries to handle out of bound frame calls or errors gracefully to recover instead of erring. Many tasks are now Threaded.
- Added Zones and Zone modules. Zones are used to define areas of interest and to trigger events.
- Optimized the ml pipeline, filtering of detected labels has more configurable options.

What
=====
Pythonic ZoneMinder wrapper
- API
- Event Server
- Logger
- Memory
- Machine Learning Modules

Documentation & Examples
=========================
Latest documentation is available <a href='https://pyzm.readthedocs.io/en/latest/'>here</a>. The documentation includes a full example.

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
- Zones
- [Amazon Rekognition support](https://medium.com/@michael-ludvig/aws-rekognition-support-for-zoneminder-object-detection-40b71f926a80) for object detection

Limitations
============
* Only for Python 3.6+
* OpenCV 4.1.1+ required for image manipulation. Available from pip (opencv-contrib-python) or self compiled with 
GPU support
* YOLO requires OpenCV 4.4+



