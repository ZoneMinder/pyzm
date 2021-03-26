"""
ZMMemory
=====================
Wrapper to access SHM for Monitor status
"""


import mmap
import struct
from collections import namedtuple
import os
from pyzm.helpers.Base import Base
import pyzm.helpers.globals as g


"""
shared_data => { type=>'SharedData', seq=>$mem_seq++, contents=> {
    size             => { type=>'uint32', seq=>$mem_seq++ }, I
    last_write_index => { type=>'uint32', seq=>$mem_seq++ }, I
    last_read_index  => { type=>'uint32', seq=>$mem_seq++ }, I
    state            => { type=>'uint32', seq=>$mem_seq++ }, I
    last_event       => { type=>'uint64', seq=>$mem_seq++ }, Q
    action           => { type=>'uint32', seq=>$mem_seq++ }, I
    brightness       => { type=>'int32', seq=>$mem_seq++ },  i
    hue              => { type=>'int32', seq=>$mem_seq++ },  i
    colour           => { type=>'int32', seq=>$mem_seq++ },  i
    contrast         => { type=>'int32', seq=>$mem_seq++ },  i
    alarm_x          => { type=>'int32', seq=>$mem_seq++ },  i
    alarm_y          => { type=>'int32', seq=>$mem_seq++ },  i
    valid            => { type=>'uint8', seq=>$mem_seq++ },  ?
    active           => { type=>'uint8', seq=>$mem_seq++ },  ?
    signal           => { type=>'uint8', seq=>$mem_seq++ },  ?
    format           => { type=>'uint8', seq=>$mem_seq++ },  ?
    imagesize        => { type=>'uint32', seq=>$mem_seq++ }, I
    epadding1        => { type=>'uint32', seq=>$mem_seq++ },  I
    startup_time     => { type=>'time_t64', seq=>$mem_seq++ },  Q
    last_write_time  => { type=>'time_t64', seq=>$mem_seq++ },  Q
    last_read_time   => { type=>'time_t64', seq=>$mem_seq++ },  Q 
    control_state    => { type=>'uint8[256]', seq=>$mem_seq++ }, s256
    alarm_cause      => { type=>'int8[256]', seq=>$mem_seq++ }, s256
"""

"""
    trigger_data => { type=>'TriggerData', seq=>$mem_seq++, 'contents'=> {
    size             => { type=>'uint32', seq=>$mem_seq++ }, I
    trigger_state    => { type=>'uint32', seq=>$mem_seq++ }, I
    trigger_score    => { type=>'uint32', seq=>$mem_seq++ }, I
    padding          => { type=>'uint32', seq=>$mem_seq++ }, I
    trigger_cause    => { type=>'int8[32]', seq=>$mem_seq++ }, s32
    trigger_text     => { type=>'int8[256]', seq=>$mem_seq++ }, s256
    trigger_showtext => { type=>'int8[256]', seq=>$mem_seq++ }, s256
  }
"""

class ZMMemory(Base):


    def __init__(self,api=None, path='/dev/shm', mid=None):
        self.api = api

        self.alarm_state_stages = {
        'STATE_IDLE':0,
        'STATE_PREALARM':1,
        'STATE_ALARM':2,
        'STATE_ALERT':3,
        'STATE_TAPE':4,
        'ACTION_GET':5,
        'ACTION_SET':6,
        'ACTION_RELOAD':7,
        'ACTION_SUSPEND':8,
        'ACTION_RESUME':9,
        'TRIGGER_CANCEL':10,
        'TRIGGER_ON':11,
        'TRIGGER_OFF':12
        }
        self.fhandle = None
        self.mhandle = None

        if not mid:
            raise ValueError ('No monitor specified')
        self.fname = path+'/zm.mmap.'+str(mid)
        self.reload()
        
    def reload(self):
        """Reloads monitor information. Call after you get
        an invalid memory report    
        
        Raises:
            ValueError: if no monitor is provided
        """
        self.close()
        self.fhandle = open(self.fname, "r+b")
        sz = os.path.getsize(self.fname)
        if not sz:
            raise ValueError ('Invalid size: {} of {}'.format(sz, self.fname))

        self.mhandle = mmap.mmap(self.fhandle.fileno(), 0, access=mmap.ACCESS_READ)
        self.sd = None
        self.td = None
        self._read()

    def is_valid(self):
        """True if the memory handle is valid
        
        Returns:
            bool: True if memory handle is valid
        """
        try:
            d = self._read()
            return not d['shared_data']['size']==0
        except Exception as e:
            g.logger.Error('Memory: {}'.format(e))
            return False

    def is_alarmed(self):
        """True if monitor is currently alarmed
        
        Returns:
            bool: True if monitor is currently alarmed
        """
        
        d = self._read()
        return int(d['shared_data']['state']) == self.alarm_state_stages['STATE_ALARM']    
    
    def alarm_state(self):
        """Returns alarm state
        
        Returns:
            dict: as below::

                {
                    'id': int # state id
                    'state': string # name of state
                }
            
        """
        
        d = self._read()
        return {
            'id': d['shared_data']['state'],
            'state': list(self.alarm_state_stages.keys())[list(self.alarm_state_stages.values()).index(int( d['shared_data']['state']))]
        }        
    
        
    def last_event(self):
        """Returns last event ID
        
        Returns:
            int: last event id
        """
        d = self._read()
        return d['shared_data']['last_event']

    def cause(self):
        """Returns alarm and trigger cause as applicable
        
        Returns:
            dict: as below::

                {
                    'alarm_cause': string 
                    'trigger_cause': string
                }
            
        """
        d=self._read()
        return {
            'alarm_cause': d['shared_data'].get('alarm_cause'), # May not be there
            'trigger_cause': d['shared_data'].get('trigger_cause'),
        }

    def trigger(self):
        """Returns trigger information
        
        Returns:
            dict: as below::

                {
                    'trigger_text': string,
                    'trigger_showtext': string,
                    'trigger_cause': string,
                    'trigger:state' {
                        'id': int,
                        'state': string
                    }
                }
        """
        
        d=self._read()
        return {
            'trigger_text': d['trigger_data'].get('trigger_text'),
            'trigger_showtext': d['trigger_data'].get('trigger_showtext'),
            'trigger_cause': d['trigger_data'].get('trigger_cause'),
            'trigger_state': {
                'id':d['trigger_data'].get('trigger_state'),
                'state': d['trigger_data']['trigger_state']
            }

        }

    def _read(self):
        self.mhandle.seek(0)
        SharedData = namedtuple('SharedData', 'size last_write_index last_read_index state last_event action brightness hue color contrast alarm_x alarm_y valid active signal format imagesize epadding1 startup_time last_write_time last_read_time control_state alarm_cause')
        s = SharedData._make(struct.unpack('IIIIQIiiiiii????IIQQQ256s256s',self.mhandle.read(600)))
        TriggerData = namedtuple('TriggerData', 'size trigger_state trigger_score padding trigger_cause trigger_text trigger_showtext')
        t = TriggerData._make(struct.unpack('IIII32s256s256s', self.mhandle.read(560)))
        self.sd = s._asdict()
        self.td = t._asdict()

        self.sd['alarm_cause'] = self.sd['alarm_cause'].split(b'\0',1)[0].decode()
        self.sd['control_state'] = self.sd['control_state'].split(b'\0',1)[0].decode()
        self.td['trigger_cause'] = self.td['trigger_cause'].split(b'\0',1)[0].decode()
        self.td['trigger_text'] = self.td['trigger_text'].split(b'\0',1)[0].decode()
        self.td['trigger_showtext'] = self.td['trigger_showtext'].split(b'\0',1)[0].decode()
        return { 'shared_data':self.sd, 'trigger_data':self.td}

    
    def get(self):
        """returns raw shared and trigger data as a dict
        
        Returns:
            dict: raw shared and trigger data ::

                {
                    'shared_data': dict, # of shared data,
                    'trigger_data': dict # trigger data          
                }
        """
        return self._read()

    def get_shared_data(self):
        """Returns just the shared data
        
        Returns:
            dict: shared data
        """
        return self._read()['shared_data']

    def get_trigger_data(self):
        """Returns just the trigger data

        
        Returns:
            dict: trigger data
        """
        return self._read()['trigger_data']

    def close(self):
        """Closes the handle
        """
        try:
            if self.mhandle: self.mhandle.close()
            if self.fhandle: self.fhandle.close()
        except Exception as e:
            pass


