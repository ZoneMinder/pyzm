"""
Module Event
============
Each Event will hold a single ZoneMinder Event.
It is basically a bunch of getters for each access to event data.
If you don't see a specific getter, just use the generic get function to get
the full object
"""

from pyzm.helpers.Base import Base

class Event(Base):
    def __init__(self, event=None, api=None, logger=None):
        Base.__init__(self, logger)
        self.event = event
        self.logger = logger
        self.api = api
        
    def get(self):
        return self.event['Event']
    
    def monitor_id(self):
        return int(self.event['Event']['MonitorId'])
    
    def id(self):
        return int(self.event['Event']['Id'])

    def name(self):
        return self.event['Event']['Name'] or None
    
    def video_file(self):
        return self.event['Event'].get('DefaultVideo')
    
    def cause(self):
        return self.event['Event']['Cause'] or None
    
    def notes(self):
        return self.event['Event']['Notes'] or None
    
    def fspath(self):
        return self.event['Event'].get('FileSystemPath')
    
    def duration(self):
        return float(self.event['Event']['Length'])
    
    def total_frames(self):
        return int(self.event['Event']['Frames'])
    
    def alarmed_frames(self):
        return int(self.event['Event']['AlarmFrames'])
    
    def score(self):
        return {
            'total': float(self.event['Event']['TotScore']),
            'average':float(self.event['Event']['AvgScore']),
            'max':float(self.event['Event']['MaxScore'])
        }