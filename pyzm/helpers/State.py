"""
State
======
Each State will hold a single ZoneMinder State.
It is basically a bunch of getters for each access to event data.
If you don't see a specific getter, just use the generic get function to get
the full object
"""
from typing import Optional

g = None


class State:
    def __init__(self, state=None, globs=None):
        global g
        g = globs
        self.state = state
        self.api = g.api
    
    def get(self):
        """Returns raw state object
        
        Returns:
            :class:`pyzm.helpers.State`: raw state object
        """
        return self.state['State']
    

    def active(self):
        """whether this state is active or not
        
        Returns:
            bool: True if active
        """
        return self.state['State']['IsActive'] == '1'

    def definition(self):
        """Returns the description text of this state
        
        Returns:
            string: description
        """
        return self.state['State']['Definition'] or None
    
    def name(self):
        """Name of this state
        
        Returns:
            string: name of this state
        """
        return self.state['State']['Name']
    
    def id(self):
        """Id of this state
        
        Returns:
            int: id of this state
        """
        return int(self.state['State']['Id'])

   
