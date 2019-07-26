"""
Module Events
=================
Holds a list of Events for a ZM configuration
You can pass different conditions to filter the events
Each invocation results in a new API call as events are very dynamic
"""

from pyzm.helpers.Base import Base
from pyzm.helpers.Event import Event
import requests
import dateparser

class Events:
    def __init__(self, logger=None, api=None, options=None):
        Base.__init__(self, logger)
        self.api = api
        self.events = []
        self._load(options)
    
    def list(self):
        return self.events


    def _load(self, options={}):
        self.logger.Debug(1,'Retrieving events via API')
        url_filter=''
       
        if options.get('from'):
            url_filter +=  '/StartTime >=:'+dateparser.parse(options.get('from')).strftime('%Y-%m-%d %H:%M:%S')
        if options.get('to'):
            url_filter+=  '/EndTime <=:'+dateparser.parse(options.get('from')).strftime('%Y-%m-%d %H:%M:%S')
        if options.get('mid'):
            url_filter+= '/MonitorId =:'+str(options.get('mid'))
        if options.get('min_alarmed_frames'):
            url_filter+='/AlarmFrames >=:'+str(options.get('min_alarmed_frames'))
        if options.get('max_alarmed_frames'):
            url_filter+='/AlarmFrames <=:'+str(options.get('max_alarmed_frames'))
        if options.get('object_only'):
            url_filter+='/Notes REGEXP:detected:'

        # catch all
        if options.get('raw_filter'):
             url_filter+=options.get('raw_filter')

        # tbd - no need for url_prefix in options
        url_prefix = options.get('url_prefix',self.api.api_url + '/events/index' )
       
        url = url_prefix + url_filter +'.json'
        params = {
            'sort':'StartTime',
            'direction': 'desc',
            'page': 1
        }
        for k in options:
            if k in params:
                params[k] = options[k]

        numevents = options.get('max_events', 100)
        params['limit'] = numevents
        currevents = 0
        self.events = []
        events= []
        while True:
            r = self.api.make_request(url=url, query=params)
            events.extend(r.get('events'))
            pagination = r.get('pagination')
            if not pagination:
                break
            if  not pagination.get('nextPage'):
                break
            currevents += int(pagination.get('current'))
            if currevents >= numevents:
                break
            params['page'] +=1 
        
        for event in events:
            self.events.append(Event(event=event,api=self.api,logger= self.logger))
    
    def get(self, options={}):
        self._load(options)
        return self.events

  
        