"""
ZMMemory
=====================
Wrapper to access SHM for Monitor status

Supports both ZoneMinder 1.36.x and 1.38.0+ SharedData struct formats.
The version is auto-detected from the 'size' field in shared memory.
"""


import mmap
import struct
from collections import namedtuple
import os
from pyzm.helpers.Base import Base
import pyzm.helpers.globals as g


# ZM 1.36.x SharedData struct format and field names
_STRUCT_FMT_136 = '@IiiIddQIiiiiii????IIIIqqqq256s256s64s64s'
_FIELDS_136 = (
    'size last_write_index last_read_index state '
    'capture_fps analysis_fps last_event action '
    'brightness hue color contrast alarm_x alarm_y '
    'valid active signal format '
    'imagesize last_frame_score audio_frequency audio_channels '
    'startup_time heartbeat_time last_write_time last_read_time '
    'control_state alarm_cause video_fifo audio_fifo'
)
_SD_STRING_FIELDS_136 = [
    'alarm_cause', 'control_state', 'audio_fifo', 'video_fifo'
]

# ZM 1.38.0+ SharedData struct format and field names
_STRUCT_FMT_138 = '@IiiiIddddQIiiiiii????????IIIIqqqqqq256s256s64s64s64s'
_FIELDS_138 = (
    'size last_write_index last_read_index image_count state '
    'capture_fps analysis_fps latitude longitude last_event action '
    'brightness hue color contrast alarm_x alarm_y '
    'valid capturing analysing recording signal format reserved1 reserved2 '
    'imagesize last_frame_score audio_frequency audio_channels '
    'startup_time heartbeat_time last_write_time last_read_time '
    'last_viewed_time last_analysis_viewed_time '
    'control_state alarm_cause video_fifo audio_fifo janus_pin'
)
_SD_STRING_FIELDS_138 = [
    'alarm_cause', 'control_state', 'audio_fifo', 'video_fifo', 'janus_pin'
]

# Pre-calculate struct sizes for version detection
_SIZE_136 = struct.calcsize(_STRUCT_FMT_136)
_SIZE_138 = struct.calcsize(_STRUCT_FMT_138)

# TriggerData struct (unchanged between versions)
_TRIGGER_FMT = 'IIII32s256s256s'
_TRIGGER_SIZE = struct.calcsize(_TRIGGER_FMT)


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
        self._zm_version = None  # Will be '1.36' or '1.38' after first read

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
        self._zm_version = None
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

    def _detect_version(self):
        """Detect ZM version from the size field in SharedData.
        
        The first uint32 in SharedData is the struct size, which differs
        between ZM versions:
        - ZM 1.36.x: 760 bytes
        - ZM 1.38.0+: 872 bytes
        """
        self.mhandle.seek(0)
        size_bytes = self.mhandle.read(4)
        size_val = struct.unpack('@I', size_bytes)[0]
        if size_val == _SIZE_138:
            self._zm_version = '1.38'
        elif size_val == _SIZE_136:
            self._zm_version = '1.36'
        else:
            # Default to 1.36 format but warn
            self._zm_version = '1.36'
            try:
                g.logger.Warning(
                    'ZMMemory: Unknown SharedData size {} in {}, '
                    'expected {} (ZM 1.36) or {} (ZM 1.38). '
                    'Falling back to ZM 1.36 format.'.format(
                        size_val, self.fname, _SIZE_136, _SIZE_138
                    )
                )
            except Exception:
                pass

    def _read(self):
        # Detect version on first read
        if self._zm_version is None:
            self._detect_version()

        self.mhandle.seek(0)

        if self._zm_version == '1.38':
            struct_fmt = _STRUCT_FMT_138
            fields = _FIELDS_138
            string_fields = _SD_STRING_FIELDS_138
        else:
            struct_fmt = _STRUCT_FMT_136
            fields = _FIELDS_136
            string_fields = _SD_STRING_FIELDS_136

        SharedData = namedtuple('SharedData', fields)
        struct_size = struct.calcsize(struct_fmt)
        s = SharedData._make(struct.unpack(struct_fmt, self.mhandle.read(struct_size)))

        TriggerData = namedtuple('TriggerData', 'size trigger_state trigger_score padding trigger_cause trigger_text trigger_showtext')
        t = TriggerData._make(struct.unpack(_TRIGGER_FMT, self.mhandle.read(_TRIGGER_SIZE)))
        self.sd = s._asdict()
        self.td = t._asdict()

        for key in string_fields:
            self.sd[key] = self.sd[key].split(b'\0',1)[0].decode(errors='replace')
        self.td['trigger_cause'] = self.td['trigger_cause'].split(b'\0',1)[0].decode(errors='replace')
        self.td['trigger_text'] = self.td['trigger_text'].split(b'\0',1)[0].decode(errors='replace')
        self.td['trigger_showtext'] = self.td['trigger_showtext'].split(b'\0',1)[0].decode(errors='replace')

        # For backward compatibility, add 'active' as alias for 'capturing' in ZM 1.38
        if self._zm_version == '1.38' and 'capturing' in self.sd:
            self.sd['active'] = self.sd['capturing']

        return {'shared_data': self.sd, 'trigger_data': self.td}


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
