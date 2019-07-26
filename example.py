import pyzm.api as zmapi
import sys,traceback
import pyzm.helpers.ZMLog as zmlog #only if you want to log to ZM

# Assuming you want to log to ZM
# You can override default ZM Log settings
# programatically
zm_log_override = {
    'log_level_syslog' : 3,
    'log_level_db': -5,
    'log_debug': 1,
    'log_level_file': -5,
    'log_debug_target': None
}


zmlog.init(name='apitest',override=zm_log_override)

api_options = {
    'apiurl': 'https://demo.zoneminder.com/zm/api',
    'user': 'zmuser',
    'password': 'zmpass',
    'logger': zmlog # We connect the API to zmlog 
    #'logger': None # use none if you don't want to log to ZM
}

# lets init the API
try:
    zmapi = zmapi.ZMApi(options=api_options)
except Exception as e:
    print ('Error: {}'.format(str(e)))
    print(traceback.format_exc())
    exit(1)

#
print ("Getting Monitors")
ms = zmapi.monitors()
for m in ms.list():
    print ('Name:{} Enabled:{} Type:{} Dims:{}'.format(m.name(), m.enabled(), m.type(), m.dimensions())) 


cam_name='DemoVirtualCam1'

event_filter = {
    'from': '20 hours ago',
    'object_only':False,
    'min_alarmed_frames': 3
}

print ("getting events across all monitors")
es = zmapi.events(event_filter)
print ('I got {} events'.format(len(es.list())))


print ('Getting events for {} with filter: {}'.format(cam_name, event_filter))

cam_events = ms.find(name=cam_name).events(options=event_filter)
for e in cam_events.list():
    print ('Event:{} Cause:{} Notes:{}'.format(e.name(), e.cause(), e.notes()))

print ('Getting states:')
states = zmapi.states()
for state in states.list():
    print ('State:{}[{}], active={}, details={}'.format(state.name(), state.id(), state.active(), state.definition()))

zmlog.close()


