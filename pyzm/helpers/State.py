"""
Module State
=================
Each State will hold a single ZoneMinder State.
It is basically a bunch of getters for each access to event data.
If you don't see a specific getter, just use the generic get function to get
the full object
"""


from pyzm.helpers.Base import Base

class State(Base):
    def __init__(self, api=None, state=None, logger=None):
        Base.__init__(self, logger)
        self.state = state
        self.api = api
    
    def get(self):
        return self.state['State']
    

    def active(self):
        return self.state['State']['IsActive'] == '1'

    def definition(self):
        return self.state['State']['Definition'] or None
    
    def name(self):
        return self.state['State']['Name']
    
    def id(self):
        return int(self.state['State']['Id'])

   
