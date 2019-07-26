"""
Module Monitor
=================
Each Monitor will hold a single ZoneMinder Monitor.
It is basically a bunch of getters for each access to event data.
If you don't see a specific getter, just use the generic get function to get
the full object
"""


from pyzm.helpers.Base import Base

class Monitor(Base):
    def __init__(self, api=None, monitor=None, logger=None):
        Base.__init__(self, logger)
        self.monitor = monitor
        self.api = api
    
    def get(self):
        return self.monitor['Monitor']

    def enabled(self):
        return self.monitor['Monitor']['Enabled'] == '1'

    def function(self):
        return self.monitor['Monitor']['Function']
    
    def name(self):
        return self.monitor['Monitor']['Name']
    
    def id(self):
        return int(self.monitor['Monitor']['Id'])

    def type(self):
        return self.monitor['Monitor']['Type']
    
    def dimensions(self):
        return { 'width':int(self.monitor['Monitor']['Width']), 
                 'height':int(self.monitor['Monitor']['Height'])
        }
    
    def events(self, options={}):
        options['mid'] = self.id()
        return self.api.events(options=options)
        

