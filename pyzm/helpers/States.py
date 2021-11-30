"""
States
=======
Holds a list of States for a ZM configuration
Given states are fairly static, maintains a cache of states
which can be overriden 
"""


from pyzm.helpers.State import State
from typing import Optional

g = None


class States:
    def __init__(self, globs=g):
        global g
        g = globs
        self.api = g.api
        self.states = []
        self._load()

    def __iter__(self):
        if self.states:
            for state in self.states:
                yield state

    def _load(self,options={}):
        g.logger.debug(2, 'Retrieving states via API')
        url = self.api.api_url +'/states.json'
        r = self.api.make_request(url=url)
        states = r.get('states')
        for state in states:
           self.states.append(State(state=state, globs=g))


    def list(self):
        """Returns list of state objects
        
        Returns:
            list of `pyzm.helpers.State`: list of state objects
        """
        return self.states

    
    def find(self, id=None, name=None):
        """Return a state object that matches either and id or a
        
        Args:
            id (int, optional): Id of state. Defaults to None.
            name (string, optional): name of state. Defaults to None.
        
        Returns:
            :class:`pyzm.helpers.State`: State object that matches
        """
        if not id and not name:
            return None
        match = None
        if id:
            key = 'Id'
        else:
            key = 'Name'
    
        for state in self.states:
            if id and state.id() == id:
                match = state
                break
            if name and state.name().lower() == name.lower():
                match = state
                break
        return match
        


    

