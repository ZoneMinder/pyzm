"""
Event
======
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
        """Returns event object
        
        Returns:
            :class:`pyzm.helpers.Event`: Event object
        """
        return self.event['Event']
    
    def monitor_id(self):
        """returns monitor ID of event object
        
        Returns:
            int: monitor id
        """
        return int(self.event['Event']['MonitorId'])
    
    def id(self):
        """returns event id of event
        
        Returns:
            int: event id
        """
        return int(self.event['Event']['Id'])

    def name(self):
        """returns name of event
        
        Returns:
            string: name of event
        """      
        return self.event['Event']['Name'] or None
    
    def video_file(self):
        """returns name of video file in which the event was stored
        
        Returns:
            string: name of video file
        """
        return self.event['Event'].get('DefaultVideo')
    
    def cause(self):
        """returns event cause
        
        Returns:
            string: event cause
        """ 
        return self.event['Event']['Cause'] or None
    
    def notes(self):
        """returns event notes
        
        Returns:
            string: event notes
        """
        return self.event['Event']['Notes'] or None
    
    def fspath(self):
        """returns the filesystem path where the event is stored. Only available in ZM 1.33+
        
        Returns:
            string: path
        """
        return self.event['Event'].get('FileSystemPath')
    
    def duration(self):
        """Returns duration of event in seconds
        
        Returns:
            float: duration
        """
        return float(self.event['Event']['Length'])
    
    def total_frames(self):
        """Returns total frames in event
        
        Returns:
            int: total frames
        """
        return int(self.event['Event']['Frames'])
    
    def alarmed_frames(self):
        """Returns total alarmed frames in event
        
        Returns:
            int: total alarmed frames
        """
        return int(self.event['Event']['AlarmFrames'])
    
    def score(self):
        """Returns total, average and max scores of event
        
        Returns:
            dict: As below::

            {
                'total': float,
                'average': float,
                'max': float
            }
        """
        return {
            'total': float(self.event['Event']['TotScore']),
            'average':float(self.event['Event']['AvgScore']),
            'max':float(self.event['Event']['MaxScore'])
        }