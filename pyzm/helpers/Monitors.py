"""
Module Monitors
=================
Holds a list of Monitors for a ZM configuration
Given monitors are fairly static, maintains a cache of monitors
which can be overriden 
"""


from pyzm.helpers.Monitor import Monitor
from pyzm.helpers.Base import Base
import requests

class Monitors(Base):
    def __init__(self,logger=None, api=None):
        Base.__init__(self, logger)
        self.api = api
        self._load()

    def _load(self,options={}):
        self.logger.Debug(1,'Retrieving monitors via API')
        url = self.api.api_url +'/monitors.json'
        r = self.api.make_request(url=url)
        ms = r.get('monitors')
        self.monitors = []
        for m in ms:
           self.monitors.append(Monitor(monitor=m,api=self.api, logger=self.logger))


    def list(self):
        return self.monitors

    
    def find(self, id=None, name=None):
        if not id and not name:
            return None
        match = None
        if id:
            key = 'Id'
        else:
            key = 'Name'
    
        for mon in self.monitors:
            if id and mon.id() == id:
                match = mon
                break
            if name and mon.name().lower() == name.lower():
                match = mon
                break
        return match
        


    

