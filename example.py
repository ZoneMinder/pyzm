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


cam_name='DemoVirtualCam1'

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

# Various getter tests

print ("--------| Getting Monitors |-----------")
ms = zmapi.monitors()
for m in ms.list():
    print ('Name:{} Enabled:{} Type:{} Dims:{}'.format(m.name(), m.enabled(), m.type(), m.dimensions()))
    print (m.status())

print ("--------| Getting Events |-----------")
print ("Getting events across all monitors")
event_filter = {
    'from': '2 hours ago',
    'object_only':False,
    'min_alarmed_frames': 3,
    'max_events':5
}

es = zmapi.events(event_filter)
print ('I got {} events'.format(len(es.list())))

print ('Getting events for {} with filter: {}'.format(cam_name, event_filter))
cam_events = ms.find(name=cam_name).events(options=event_filter)
for e in cam_events.list():
    print ('Event:{} Cause:{} Notes:{}'.format(e.name(), e.cause(), e.notes()))

print ("--------| Getting ZM States |-----------")
states = zmapi.states()
for state in states.list():
    print ('State:{}[{}], active={}, details={}'.format(state.name(), state.id(), state.active(), state.definition()))

print ("--------| Setting Monitors |-----------")
m = ms.find(name=cam_name)
try:
    old_function = m.function()
    input ('Going to change state of {}[{}] to Monitor from {}'.format(m.name(),m.id(), old_function))
    print (m.set_parameter(options={'function':'Monitor'}))
    input ('Switching back to {}'.format(old_function))
    print (m.set_parameter(options={'function':old_function}))
except Exception as e:
    print ('Error: {}'.format(str(e)))

print ("--------| Setting Alarms |-----------")
try:
    input ('Arming {}, press enter'.format(m.name()))
    print (m.arm())
    input ('Disarming {}, press enter'.format(m.name()))
    print (m.disarm())
except Exception as e:
    print ('Error: {}'.format(str(e)))


print ("--------| Setting States |-----------")
try:
    input ('Stopping ZM press enter')
    print (zmapi.stop())
    input ('Starting ZM press enter')
    print (zmapi.start())
    for idx,state in enumerate(states.list()):
        print ('{}:{}'.format(idx,state.name()))
    i=int(input('enter state number to switch to:'))
    name = states.list()[i].name()
    print ('Changing state to: {}'.format(name))
    print (zmapi.set_state(state=name))
except Exception as e:
    print ('Error: {}'.format(str(e)))


print ("--------| Configs |-----------")
try:
    conf = zmapi.configs()
    print (conf.find(name='ZM_AUTH_HASH_LOGINS'))
except Exception as e:
    print ('Error: {}'.format(str(e)))

zmlog.close()


