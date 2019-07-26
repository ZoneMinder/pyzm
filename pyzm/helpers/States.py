"""
Module States
=================
Holds a list of States for a ZM configuration
Given states are fairly static, maintains a cache of states
which can be overriden 
"""


from pyzm.helpers.State import State
from pyzm.helpers.Base import Base
import requests

class States(Base):
    def __init__(self,logger=None, api=None):
        Base.__init__(self, logger)
        self.api = api
        self._load()

    def _load(self,options={}):
        self.logger.Debug(1,'Retrieving states via API')
        url = self.api.api_url +'/states.json'
        r = self.api.make_request(url=url)
        states = r.get('states')
        self.states = []
        for state in states:
           self.states.append(State(state=state,api=self.api, logger=self.logger))


    def list(self):
        return self.states

    
    def find(self, id=None, name=None):
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
        


    

