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


# ---------------------------------------------------------------------------
# SharedData struct layouts — each entry is (field_name, struct_format_char).
# Format chars ending with 's' (e.g. '256s') denote byte-string fields that
# need null-termination + decode.
# ---------------------------------------------------------------------------

# ZM <= 1.36
_LAYOUT_760 = [
    ('size', 'I'), ('last_write_index', 'i'), ('last_read_index', 'i'),
    ('state', 'I'),
    ('capture_fps', 'd'), ('analysis_fps', 'd'),
    ('last_event', 'Q'), ('action', 'I'),
    ('brightness', 'i'), ('hue', 'i'), ('color', 'i'), ('contrast', 'i'),
    ('alarm_x', 'i'), ('alarm_y', 'i'),
    ('valid', '?'), ('active', '?'), ('signal', '?'), ('format', '?'),
    ('imagesize', 'I'), ('last_frame_score', 'I'),
    ('audio_frequency', 'I'), ('audio_channels', 'I'),
    ('startup_time', 'q'), ('heartbeat_time', 'q'),
    ('last_write_time', 'q'), ('last_read_time', 'q'),
    ('control_state', '256s'), ('alarm_cause', '256s'),
    ('video_fifo', '64s'), ('audio_fifo', '64s'),
]

# ZM 1.38+
_LAYOUT_872 = [
    ('size', 'I'), ('last_write_index', 'i'), ('last_read_index', 'i'),
    ('image_count', 'i'), ('state', 'I'),
    ('capture_fps', 'd'), ('analysis_fps', 'd'),
    ('latitude', 'd'), ('longitude', 'd'),
    ('last_event', 'Q'), ('action', 'I'),
    ('brightness', 'i'), ('hue', 'i'), ('color', 'i'), ('contrast', 'i'),
    ('alarm_x', 'i'), ('alarm_y', 'i'),
    ('valid', '?'), ('capturing', '?'), ('analysing', '?'),
    ('recording', '?'), ('signal', '?'), ('format', '?'),
    ('reserved1', '?'), ('reserved2', '?'),
    ('imagesize', 'I'), ('last_frame_score', 'I'),
    ('audio_frequency', 'I'), ('audio_channels', 'I'),
    ('startup_time', 'q'), ('heartbeat_time', 'q'),
    ('last_write_time', 'q'), ('last_read_time', 'q'),
    ('last_viewed_time', 'q'), ('last_analysis_viewed_time', 'q'),
    ('control_state', '256s'), ('alarm_cause', '256s'),
    ('video_fifo', '64s'), ('audio_fifo', '64s'), ('janus_pin', '64s'),
]


def _build_struct_info(layout):
    """Derive format string, field names, and string-field list from a layout."""
    fmt = '@' + ''.join(char for _, char in layout)
    fields = [name for name, _ in layout]
    string_fields = [name for name, char in layout if char.endswith('s')]
    return fmt, fields, string_fields


# Registry keyed by struct size — adding a new ZM version is just one more
# _LAYOUT_xxx list appended here.
_REGISTRY = {}
for _layout in [_LAYOUT_760, _LAYOUT_872]:
    _fmt, _fields, _str_fields = _build_struct_info(_layout)
    _REGISTRY[struct.calcsize(_fmt)] = (_fmt, _fields, _str_fields)

# TriggerData — unchanged across ZM versions
_TRIGGER_FMT = 'IIII32s256s256s'
_TRIGGER_FIELDS = ['size', 'trigger_state', 'trigger_score', 'padding',
                   'trigger_cause', 'trigger_text', 'trigger_showtext']
_TRIGGER_SIZE = struct.calcsize(_TRIGGER_FMT)
_TRIGGER_STRING_FIELDS = ['trigger_cause', 'trigger_text', 'trigger_showtext']


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
        self._layout = None  # cached (fmt, fields, string_fields) after detection

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
        self._layout = None
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

    def _detect_layout(self):
        """Detect the SharedData layout from the size field (first uint32)."""
        self.mhandle.seek(0)
        size_val = struct.unpack('@I', self.mhandle.read(4))[0]
        if size_val in _REGISTRY:
            self._layout = _REGISTRY[size_val]
        else:
            raise ValueError(
                'Unknown SharedData size {} in {}. '
                'Expected one of: {}'.format(
                    size_val, self.fname, sorted(_REGISTRY.keys())
                )
            )

    def _read(self):
        if self._layout is None:
            self._detect_layout()

        struct_fmt, fields, string_fields = self._layout

        self.mhandle.seek(0)
        struct_size = struct.calcsize(struct_fmt)
        SharedData = namedtuple('SharedData', fields)
        s = SharedData._make(struct.unpack(struct_fmt, self.mhandle.read(struct_size)))

        TriggerData = namedtuple('TriggerData', _TRIGGER_FIELDS)
        t = TriggerData._make(struct.unpack(_TRIGGER_FMT, self.mhandle.read(_TRIGGER_SIZE)))

        self.sd = s._asdict()
        self.td = t._asdict()

        for key in string_fields:
            self.sd[key] = self.sd[key].split(b'\0',1)[0].decode(errors='replace')
        for key in _TRIGGER_STRING_FIELDS:
            self.td[key] = self.td[key].split(b'\0',1)[0].decode(errors='replace')

        # Backward compat: ZM 1.38 renamed 'active' to 'capturing'
        if 'capturing' in self.sd:
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
