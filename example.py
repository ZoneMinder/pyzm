import pyzm.api as zmapi
import getpass
import traceback

has_zmes = False
has_zmlog = False

try:
    import pyzm.ZMLog as zmlog #only if you want to log to ZM
    has_zmlog = True
except ImportError as e:
    print ('Could not import ZMLog, function will be disabled:'+str(e))
    zmlog = None
    

try:
    from pyzm.ZMEventNotification import ZMEventNotification as ZMES
    has_zmes = True
except ImportError as e:
    print ('Could not import ZMEventNotification, function will be disabled:'+str(e))
   

def on_es_message(msg):
    print (f'======> APP GOT MESSAGE FROM ES: {msg}')


def on_es_error(err):
    print (f'======> APP GOT ERROR  FROM ES: {err}') 

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

if has_zmlog:
    zmlog.init(name='apitest',override=zm_log_override)

if has_zmes:
    i = input ('Test the Event Server? [y/N]')
    if i=='y':
        ES_URL=None
        ES_USER=None
        ES_PASSWORD=None
        ALLOW_UNTRUSTED=False
        
        if not ES_URL: ES_URL = input ('Enter ES URL (example wss://foo:9000):')
        if not ES_USER: ES_USER = input ('Enter ES user (example admin):')
        if not ES_PASSWORD: ES_PASSWORD = getpass.getpass('Enter ES password:')

        es = ZMES({
            'url':ES_URL,
            'password': ES_PASSWORD,
            'user': ES_USER,
            'allow_untrusted': ALLOW_UNTRUSTED,
            'logger': zmlog,
            'on_es_message': on_es_message,
            'on_es_error': on_es_error
                
        })
        # send a legit command
        es.send({"event":"control","data":{"type":"filter","monlist":"1,2,5,6,7,8,9,10", "intlist":"0,0,0,0,0,0,0,0"}})

        # send a bad command
        es.send ('WHAT THE!')

        input ('press a key to proceed with the rest...')

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
    'from': '2 hours ago', # this will use localtimezone, use 'tz' for other timezones
    'object_only':False,
    'min_alarmed_frames': 3,
    'max_events':5,
    
}

es = zmapi.events(event_filter)
print ('I got {} events'.format(len(es.list())))

input ("Now revoke the access token in ZM and I'll try the same api again. Press ENTER when ready....(only applies to ZM 1.33+)")
es = zmapi.events(event_filter)
print ('repeat API - I got {} events'.format(len(es.list())))
input ('press ENTER to continue')

print ('Getting events for {} with filter: {}'.format(cam_name, event_filter))
cam_events = ms.find(name=cam_name).events(options=event_filter)
for e in cam_events.list():
    print ('Event:{} Cause:{} Notes:{}'.format(e.name(), e.cause(), e.notes()))

print ('Getting event summaries')
m = ms.find(name=cam_name)

# These will use server timezone
print ('Monitor {} has {} events {}'.format(m.name(), m.eventcount(options={'from':'1 hour ago','tz': zmapi.tz()}), '1 hour ago'))
print ('Monitor {} has {} events {}'.format(m.name(), m.eventcount(options={'from':'1 day ago','tz': zmapi.tz()}), '1 day ago'))

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


