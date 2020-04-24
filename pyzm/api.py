"""
ZMApi
=============
Python API wrapper for ZM.
Exposes login, monitors, events, etc. API
"""

import requests
import datetime
from pyzm.helpers.Base import Base
from pyzm.helpers.Monitors import Monitors
from pyzm.helpers.Events import Events
from pyzm.helpers.States import States
from pyzm.helpers.Configs import Configs



class ZMApi (Base):
    def __init__(self,options={}):
        '''
        Options is a dict with the following keys:

            - apiurl - the full API URL (example https://server/zm/api)
            - portalurl - the full portal URL (example https://server/zm). Only needed if you are downloading events/images
            - user - username
            - password - password
            - disable_ssl_cert_check - if True will let you use self signed certs
            - logger - (OPTIONAL) function used for logging. If none specified, a simple logger will be used that prints to console. You could instantiate and connect the :class:`pyzm.helpers.ZMLog` module here if you want to use ZM's logging.

            Note: you can connect your own customer logging class to the API in which case all modules will use your custom class. Your class will need to implement some methods for this to work. See :class:`pyzm.helpers.Base.SimpleLog` for method details.
        '''
        super().__init__(options.get('logger'))
        self.api_url = options.get('apiurl')
        self.portal_url = options.get('portalurl')
        if not self.portal_url and self.api_url.endswith('/api'):
            self.portal_url = self.api_url[:-len('/api')] 
            self.logger.Debug (1,'Guessing portal URL is: {}'.format(self.portal_url))

        self.options = options
        
        self.authenticated = False
        self.access_token = ''
        self.refresh_token = ''
        self.access_token_expires = None
        self.refresh_token_expires = None
        self.refresh_token_datetime = None
        self.legacy_credentials = None
        self.session = requests.Session()
        if options.get('disable_ssl_cert_check'):
            self.session.verify = False
            self.logger.Debug (1, 'Warning, SSL certificate check has been disbled')
            from urllib3.exceptions import InsecureRequestWarning
            requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
        self.api_version = None
        self.zm_version = None
        self.zm_tz = None
        
        self._login()
        
        self.Monitors = Monitors(logger=options.get('logger'),api=self)
        self.Events = None
        self.Configs = Configs(logger=options.get('logger'), api=self)

    def _versiontuple(self,v):
        #https://stackoverflow.com/a/11887825/1361529
        return tuple(map(int, (v.split("."))))

    def version(self):
        """Returns version of API and ZM
        
        Returns:
            dict: Version of API and ZM::

            {
                status: string # if 'error' then will also have 'reason' 
                api_version: string # if status is 'ok'
                zm_version: string # if status is 'ok'
            }
        """
        if not self.authenticated:
            return {'status':'error', 'reason':'not authenticated'}
        return {
            'status': 'ok',
            'api_version': self.api_version,
            'zm_version': self.zm_version
        }

    def tz(self):
        """Returns timezone of ZoneMinder server
       
        Returns:
           string: timezone of ZoneMinder server (or None if API not supported)
        """
        return self.zm_tz

    def authenticated(self):
        """True if login API worked
        
        Returns:
            boolean -- True if Login API worked
        """
        return self.authenticated

    def _relogin(self):
        """ Used for 401. I could use _login too but decided to do a simpler fn
        """
        url = self.api_url+'/host/login.json'
        if self._versiontuple(self.api_version) >= self._versiontuple('2.0'):
            # use tokens
            tr = (self.refresh_token_datetime - datetime.datetime.now()).total_seconds()


            if (tr >= 60 * 5): # 5 mins grace
                self.logger.Debug(1, 'Going to use refresh token as it still has {} minutes remaining'.format(tr/60))
                self.options['token'] = self.refresh_token
            else:
                self.logger.Debug (1, 'Refresh token only has {}s of lifetime, going to use user/pass'.format(tr))
                self.options['token'] = None
        self._login()

    def _login(self):
        """This is called by the constructor. You are not expected to call this directly.
        
        Raises:
            err: reason for failure
        """
        try:
            url = self.api_url+'/host/login.json'
            if self.options.get('token'):
                self.logger.Debug(1,'Using token for login [{}]'.format(self.options.get('token')))
                data = {'token':self.options['token']}
            else:
                self.logger.Debug (1,'using username/password for login')
                data={'user': self.options['user'],
                    'pass': self.options['password']
                }

            r = self.session.post(url, data=data)
            if r.status_code == 401 and self.options['token']:
                self.logger.Debug (1, 'Token auth with refresh failed. Likely revoked, doing u/p login')
                self.options['token'] = None
                data={'user': self.options['user'],
                    'pass': self.options['password']
                }
                r = self.session.post(url, data=data)
                r.raise_for_status()
            else:
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
                self.refresh_token_datetime = datetime.datetime.now() + datetime.timedelta(seconds = self.refresh_token_expires)
                self.logger.Debug (1, 'Refresh token expires on:{} [{}s]'.format(self.refresh_token_datetime, self.refresh_token_expires))
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
            r = self._make_request(url)
            self.zm_tz = r.get('tz')
        except requests.exceptions.HTTPError as err:
            self.logger.Error ('Timezone API not found, relative timezones will be local time')

    def get_auth(self):
        if self._versiontuple(self.api_version) >= self._versiontuple('2.0'):
            return 'token='+self.access_token
        else:
            return self.legacy_credentials




    def _make_request(self, url=None, query={}, payload={}, type='get', reauth=True):
        type = type.lower()
        if self._versiontuple(self.api_version) >= self._versiontuple('2.0'):
            query['token'] = self.access_token
            # ZM 1.34 API bug, will be fixed soon
            # self.session = requests.Session()
    
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
            self.logger.Debug(1, 'Got API access error: {}'.format(err), 'error')
            if err.response.status_code == 401 and reauth:
                self.logger.Debug (1, 'Retrying login once')
                self._relogin()
                self.logger.Debug (1,'Retrying failed request')
                return self._make_request(url, query, payload, type, reauth=False)
            else:
                raise err


    def monitors(self, options={}):
        """Returns list of monitors. Given monitors are fairly static, maintains a cache and returns from cache on subsequent calls.
                
            Args:
                options (dict, optional): Available fields::
            
                    {
                        'force_reload': boolean # if True refreshes monitors 

                    }
            
        Returns:
            list of :class:`pyzm.helpers.Monitor`: list of monitors 
        """
        if options.get('force_reload') or not self.Monitors:
            self.Monitors = Monitors(logger=self.logger,api=self)
        return self.Monitors

    def events(self,options={}):
        """Returns list of events based on filter criteria. Note that each time you called events, a new HTTP call is made.
        
        Args:
            options (dict, optional): Various filters that will be applied to events. Defaults to {}. Available fields::
        
                {
                    'tz': string # long form timezone (example America/New_York),
                    'from': string # minimum start time (including human readable
                                   # strings like '1 hour ago')
                    'to': string # maximum end time 
                    'mid': int # monitor id
                    'min_alarmed_frames': int # minimum alarmed frames
                    'max_alarmed_frames': int # maximum alarmed frames
                    'object_only': boolean # if True will only pick events 
                                           # that have objects

                }
        
        Returns:
            list of :class:`pyzm.helpers.Event`: list of events that match criteria
        """
        self.Events = Events(logger=self.logger,api=self, options=options)
        return self.Events

    def states(self, options={}):
        """Returns configured states
        
        Args:
            options (dict, optional): Not used. Defaults to {}.
        
        Returns:
            list of  :class:`pyzm.helpers.State`: list of states
        """
        self.States = States(logger=self.logger,api=self)
        return self.States
    
    def restart(self):
        """Restarts ZoneMinder
        
        Returns:
            json: json value of restart command
        """
        return self.set_state(state='restart')
    
    def stop(self):
        """Stops ZoneMinder
        
        Returns:
            json: json value of stop command
        """
        return self.set_state(state='stop')
    
    def start(self):
        """Starts ZoneMinder
        
        Returns:
            json: json value of start command
        """
        return self.set_state(state='start')
    
    def set_state(self, state):
        """Sets Zoneminder state to specific state 
        
        Args:
            state (string): Name of state    
        
        Returns:
            json: value of state change command
        """
        if not state:
            return
        url = self.api_url +'/states/change/{}.json'.format(state)
        return self._make_request(url=url)
    
    def running(self):
        """Determines if ZoneMinder is running
        
        Returns:
            json: json value of daemonCheck query
        """
        url = self.api_url + '/host/daemonCheck.json'
        return self._make_request(url=url)
    
    def configs(self, options={}):
        """Returns config values of ZM
        
            Args:
                options (dict, optional): Defaults to {}.
                options::

                    {
                        'force_reload': boolean # if True, reloads  
                    }
        
        Returns:
            :class:`pyzm.helpers.Configs`: ZM configs
        """
        if options.get('force_reload') or not self.Configs:
            self.Configs = Configs(logger=self.logger,api=self)
        return self.Configs
