"""
Events
=======
Holds a list of Events for a ZM configuration
You can pass different conditions to filter the events
Each invocation results in a new API call as events are very dynamic

You are not expected to use this module directly. It is instantiated when you use the :meth:`pyzm.api.ZMApi.events` method of :class:`pyzm.api`
"""
from typing import Optional

from pyzm.helpers.Event import Event
import dateparser
g = None

class Events:
    def __init__(self, options=None, globs=None):
        global g
        g = globs
        self.api = g.api
        self.events = []
        self.pagination = {}
        self._load(options)

    def __len__(self):
        if self.events:
            return len(self.events)
        else:
            return 0

    def __str__(self) -> Optional[str]:
        if self.events:
            ret_val = []
            for event in self.events:
                ret_val.append(str(event))
            return str(ret_val)
        else:
            return None

    def __iter__(self):
        if self.events:
            for event in self.events:
                yield event

    def list(self) -> []:
        """Returns list of event
        
        Returns:
            list -- events of :class:`.Event`
        """
        return self.events

    def count(self) -> int:
        """Returns number of events retrieved in previous invocation
        
        Returns:
            int -- number of events
        """
        return int(self.pagination.get('count'))

    def _load(self, options=None):
        if options is None:
            options = {}
        g.logger.info('Retrieving events via API')
        url_filter = ''
        tz = {}

        if options.get('event_id'):
            url_filter += '/Id=:' + str(options.get('event_id'))
        if options.get('tz'):
            tz = {'TIMEZONE': options.get('tz')}
            # print ('USING ',tz)
        if options.get('from'):
            from_list = options.get('from').split(" to ", 1)
            if len(from_list) == 2:
                from_start = dateparser.parse(from_list[0], settings=tz)
                from_end = dateparser.parse(from_list[1], settings=tz)
                if from_start > from_end:
                    from_start, from_end = from_end, from_start

                url_filter += '/StartTime >=:' + from_start.strftime('%Y-%m-%d %H:%M:%S')
                url_filter += '/StartTime <=:' + from_end.strftime('%Y-%m-%d %H:%M:%S')
            else:
                url_filter += '/StartTime >=:' + dateparser.parse(from_list[0], settings=tz).strftime(
                    '%Y-%m-%d %H:%M:%S')
        if options.get('to'):
            to_list = options.get('to').split(" to ", 1)
            if len(to_list) == 2:
                to_start = dateparser.parse(to_list[0], settings=tz)
                to_end = dateparser.parse(to_list[1], settings=tz)
                if to_start > to_end:
                    to_start, to_end = to_end, to_start
                url_filter += '/EndTime <=:' + to_end.strftime('%Y-%m-%d %H:%M:%S')
                url_filter += '/EndTime >=:' + to_start.strftime('%Y-%m-%d %H:%M:%S')
            else:
                url_filter += '/EndTime <=:' + dateparser.parse(to_list[0], settings=tz).strftime('%Y-%m-%d %H:%M:%S')
        if options.get('mid'):
            url_filter += '/MonitorId =:' + str(options.get('mid'))
        if options.get('min_alarmed_frames'):
            url_filter += '/AlarmFrames >=:' + str(options.get('min_alarmed_frames'))
        if options.get('max_alarmed_frames'):
            url_filter += '/AlarmFrames <=:' + str(options.get('max_alarmed_frames'))
        if options.get('object_only'):
            url_filter += '/Notes REGEXP:detected:'  # 'detected' is the key for grabbing notes from DB and the zm_event_start/end wrappers

        # catch all
        if options.get('raw_filter'):
            url_filter += options.get('raw_filter')
        # print ('URL filter: ',url_filter)
        # todo - no need for url_prefix in options
        url_prefix = options.get('url_prefix', f'{self.api.api_url}/events/index')

        url = f'{url_prefix}{url_filter}.json'
        params = {
            'sort': 'StartTime',
            'direction': 'desc',
            'page': 1
        }
        for k in options:
            if k in params:
                params[k] = options[k]

        num_events = 100
        if options.get('max_events'):
            num_events = options.get('max_events')
        if options.get('limit'):
            num_events = options.get('limit')

        params['limit'] = num_events
        curr_events = 0
        self.events = []
        events = []
        while True:
            try:
                r = self.api.make_request(url=url, query=params)
            except Exception as ex:
                g.logger.error(f"Events: error making request for events -> {url}")
                raise ex
            else:
                events.extend(r.get('events'))
                pagination = r.get('pagination')
                self.pagination = pagination
                if not pagination or not pagination.get('nextPage'):
                    break
                curr_events += int(pagination.get('current'))
                if curr_events >= num_events:
                    break
                params['page'] += 1

        for event in events:
            self.events.append(Event(event=event, globs=g))

    def get(self, options=None):
        """Returns the full list of events. Typically, useful if you need access to data for which you don't have an easy getter
        
        Keyword Arguments:
        
        - options: dict with same parameters as the one you pass in :meth:`pyzm.api.ZMApi.events`. This is really a convenience instead of re-creating the instance.
        Returns:
            list -- of :class:`pyzm.helpers.Events`
        """
        if options is None:
            options = {}
        self._load(options)
        return self.events
