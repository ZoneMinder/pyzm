"""
Module Api
==========
Python API wrapper for ZM.
Exposes login, monitors, events, etc. API
"""

import requests
from pyzm.helpers.Base import Base
from pyzm.helpers.Monitors import Monitors
from pyzm.helpers.Events import Events
from pyzm.helpers.States import States
from pyzm.helpers.Configs import Configs


class ZMApi (Base):
    def __init__(self,options={}):
        Base.__init__(self, options.get('logger'))
        self.api_url = options.get('apiurl')
        self.options = options
        
        self.authenticated = False
        self.access_token = ''
        self.refresh_token = ''
        self.access_token_expires = None
        self.refresh_token_expires = None
        self.legacy_credentials = None
        self.session = requests.Session()
        self.api_version = None
        self.zm_version = None
        self.zm_tz = None
        
        self.login()
        
        self.Monitors = Monitors(logger=options.get('logger'),api=self)
        self.Events = None
        self.Configs = Configs(logger=options.get('logger'), api=self)

    def _versiontuple(self,v):
        #https://stackoverflow.com/a/11887825/1361529
        return tuple(map(int, (v.split("."))))

    def version(self):
        if not authenticated:
            return {'status':'error', 'reason':'not authenticated'}
        return {
            'status': 'ok',
            'api_version': self.api_version,
            'zm_version': self.zm_version
        }

    def tz(self):
        return self.zm_tz

    def authenticated(self):
        return self.authenticated

    def login(self):
        try:
            url = self.api_url+'/host/login.json'
            if self.options.get('token'):
                self.logger.Debug(1,'Using token for login')
                data = {'token':self.options['token']}
            else:
                self.logger.Debug (1,'using username/password for login')
                data={'user': self.options['user'],
                    'pass': self.options['password']
                }

            r = self.session.post(url, data=data)
            r.raise_for_status()
            rj = r.json()
            self.api_version = rj.get('apiversion')
            self.zm_version = rj.get('version')
            if (self._versiontuple(self.api_version) >= self._versiontuple('2.0')):
                self.logger.Debug(1,'Using new token API')
                self.access_token = rj.get('access_token','')
                self.refresh_token = rj.get('refresh_token','')
                self.access_token_expires = int(rj.get('access_token_expires'))
                self.refresh_token_expires = int(rj.get('refresh_token_expires'))
            else:
                self.logger.Info('Using old credentials API. Recommended you upgrade to token API')
                self.legacy_credentials = rj.get('credentials')
                if (rj.get('append_password') == '1'):
                    self.legacy_credentials = self.legacy_credentials + self.options['password']
            self.authenticated = True
            #print (vars(self.session))

        except requests.exceptions.HTTPError as err:
            self.logger.Error('Got API login error: {}'.format(err), 'error')
            self.authenticated = False
            raise err

        # now get timezone
        url = self.api_url + '/host/gettimezone.json'
        
        try:
            r = self.make_request(url)
            self.zm_tz = r.get('tz')
        except requests.exceptions.HTTPError as err:
            self.logger.Error ('Timezone API not found, relative timezones will be local time')
        

    def make_request(self, url=None, query={}, payload={}, type='get'):
        type = type.lower()
        if self._versiontuple(self.api_version) >= self._versiontuple('2.0'):
            query['token'] = self.access_token
            # ZM 1.34 API bug, will be fixed soon
            self.session = requests.Session()
    
        else:
            # credentials is already query formatted
            lurl = url.lower()
            if lurl.endswith('json') or lurl.endswith('/'):
                qchar = '?'
            else:
                qchar = '&'
            url += qchar + self.legacy_credentials
            
        try:
            self.logger.Debug(1,'make_request called with url={} payload={} type={} query={}'.format(url,payload,type,query))
            if type=='get':
                r = self.session.get(url, params=query)
            elif type=='post':
                r = self.session.post(url, data=payload, params=query)
            elif type=='put':
                r = self.session.put(url, data=payload, params=query)
            elif type=='delete':
                r = self.session.delete(url, data=payload, params=query)
            else:
                self.logger.Error('Unsupported request type:{}'.format(type))
                raise ValueError ('Unsupported request type:{}'.format(type))
            #print (url, params)
            #r = requests.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as err:
            self.logger.Error('Got API access error: {}'.format(err), 'error')
            raise err


    def monitors(self, options={}):
        if options.get('force_reload') or not self.Monitors:
            self.Monitors = Monitors(logger=self.logger,api=self)
        return self.Monitors

    def events(self,options={}):
        self.Events = Events(logger=self.logger,api=self, options=options)
        return self.Events

    def states(self, options={}):
        self.States = States(logger=self.logger,api=self)
        return self.States
    
    def restart(self):
        return self.set_state(state='restart')
    
    def stop(self):
        return self.set_state(state='stop')
    
    def start(self):
        return self.set_state(state='start')
    
    def set_state(self, state=None):
        if not state:
            return
        url = self.api_url +'/states/change/{}.json'.format(state)
        return self.make_request(url=url)

    def configs(self, options={}):
        if options.get('force_reload') or not self.Configs:
            self.Configs = Monitors(logger=self.logger,api=self)
        return self.Configs
