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
    
    def set_parameter(self, options={}):
        url = self.api.api_url+'/monitors/{}.json'.format(self.id())
        payload = {}
        if options.get('function'):
            payload['Monitor[Function]'] = options.get('function')
        if options.get('name'):
            payload['Monitor[Name]'] = options.get('name')
        if options.get('enabled'):
            enabled = '1' if options.get('enabled') else '0'
            payload['Monitor[Enabled]'] = enabled

        if options.get('raw'):
            for k in options.get('raw'):
                payload[k] = options.get('raw')[k]
               
        if payload:
            return self.api.make_request(url=url, payload=payload, type='post')

    def arm(self):
        return self._set_alarm(type='on')

    def disarm(self):
        return self._set_alarm(type='off')

    def _set_alarm(self,type='on'):
        url = self.api.api_url+'/monitors/alarm/id:{}/command:{}.json'.format(self.id(), type)
        return self.api.make_request(url=url)



    def status(self):
        url = self.api.api_url+'/monitors/daemonStatus/id:{}/daemon:zmc.json'.format(self.id())
        return self.api.make_request(url=url)
