### What

ZoneMinder Python API and Logger

### Limitations
* Only for Python3
* Basic support for now

Current modules:
* Logger
* API
  * Monitors
  * Events
  * States

### Usage

At a high level, you have a ZonemMinder logging module in python and an evolving ZM API wrapper in python.
You can mix and match them, for example:
- You only want to use ZM Logging for your Python app
- You only want to use ZM APIs in your Python app but don't want to use ZM Logging. This is typically the case if say, you want to develop an app that needs ZM APIs but you want to use a different logging system, such as when you are developing a HomeAssistant component that talks to ZoneMinder (in this case, you'd want to use HomeAssistant's logging system)
- You want to use ZM APIs and use ZM's logging system. This is typically the case if you are developing a python component that directly ties into ZM. An example if this is the machine learning hook system of ZMEventserver

### Example

Take a look at <a href='https://github.com/pliablepixels/pyzm/blob/master/example.py'>the example</a> for a good staring point.

