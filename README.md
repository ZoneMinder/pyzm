# *** This is the 'neo-ZMES' forked version! ***
## PULL FROM pull_req branch to test the code that is being reviewed for merge to upstream ZoneMinder repo
== I am Using GitHub Under Protest ==

This project is currently hosted on GitHub.  This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Souce Software
(FOSS).  I am deeply concerned about using a proprietary system like GitHub
to develop my FOSS project. I urge you to read about the
[Give up GitHub](https://GiveUpGitHub.org) campaign from
[the Software Freedom Conservancy](https://sfconservancy.org) to understand
some of the reasons why GitHub is not a good place to host FOSS projects.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission.  We do not consent to GitHub's use of this project's
code in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

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



