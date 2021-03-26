import pyzm
import pyzm.api as zmapi
import getpass
import traceback
import pyzm.ZMMemory as zmmemory
import time
import pyzm.helpers.globals as g

use_zmlog = True
use_zmes = True



has_zmes = False
has_zmlog = False

print ('Using pyzm version: {}'.format(pyzm.__version__))
if use_zmlog:
    try:
        import pyzm.ZMLog as zmlog #only if you want to log to ZM
        has_zmlog = True
    except ImportError as e:
        print ('Could not import ZMLog, function will be disabled:'+str(e))
        zmlog = None
        

if use_zmes:
    try:
        from pyzm.ZMEventNotification import ZMEventNotification as ZMES
        has_zmes = True
    except ImportError as e:
        print ('Could not import ZMEventNotification, function will be disabled:'+str(e))
    

def on_es_message(msg):
    print (f'======> APP GOT MESSAGE FROM ES: {msg}')


def on_es_error(err):
    print (f'======> APP GOT ERROR  FROM ES: {err}') 


# ----------------- MAIN -------------------------
# Assuming you want to log to ZM
# You can override default ZM Log settings
# programatically
zm_log_override = {
    'log_level_syslog' : 3,
    'log_level_db': -5,
    'log_debug': 1,
    #'log_level_file': -5,
    'log_debug_target': None
}

if has_zmlog:
    zmlog.init(name='apitest',override=zm_log_override)
    print ("Log inited")
   

i = input ('Try machine learning tests? [y/N]').lower()
if i == 'y':

    import cv2
    fname = input ('Enter full path to image file:')
    if fname:
        print (f'Reading {fname}')
        img = cv2.imread(fname)
    else:
        print (f'Reading /tmp/image.jpg')
        img = cv2.imread('/tmp/image.jpg')

    i = input ('Try TPU tests? [y/N]').lower()
    if i == 'y':
        options = {
            'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
            'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
            'object_min_confidence': 0.3
        }
        import pyzm.ml.coral_edgetpu as tpu
        m = tpu.Tpu(options=options)
        b,l,c = m.detect(img)
        print (b,l,c)

    i = input ('Try OpenCV tests? [y/N]').lower()
    if i == 'y':
        options = {
            'object_weights':'/var/lib/zmeventnotification/models/yolov4/yolov4.weights',
            'object_labels': '/var/lib/zmeventnotification/models/yolov4/coco.names',
            'object_config': '/var/lib/zmeventnotification/models/yolov4/yolov4.cfg',
            'object_processor': 'cpu',
            'object_min_confidence': 0.3
        }
        import pyzm.ml.yolo as yolo
        m = yolo.Yolo(options=options)
    
        b,l,c = m.detect(img)
        print (b,l,c)

    i = input ('Try Face recognition tests? [y/N]').lower()
    if i == 'y':
        options = {
            'known_images_path': '/var/lib/zmeventnotification/known_faces',
            'face_recog_dist_threshold':0.6,
            'unknown_face_name':'klingon',
            'save_unknown_faces':'no',
            'save_unknown_faces_leeway_pixels':100,
            'unknown_images_path':'/var/lib/zmeventnotification/unknown_faces',
            'face_detection_framework': 'dlib',
            'face_recognition_framework': 'dlib',


        }
        import pyzm.ml.face as face
        m = face.Face(options=options)
        b,l,c = m.detect(img)
        print (b,l,c)

cam_name='DemoVirtualCam1'
api_options = {
    'apiurl': 'https://demo.zoneminder.com/zm/api',
    'portalurl': 'https://demo.zoneminder.com/zm',
    'user': 'zmuser',
    'password': 'zmpass',
    #'disable_ssl_cert_check': True
}



print ('Running examples on {}'.format(api_options[
    'apiurl'
]))

i = input ('Try monitor shared memory tests? [y/N]').lower()
if i == 'y':
    mid = int(input ('Enter monitor ID:'))
    while True:
        k = 'y'
        try:
            m = zmmemory.ZMMemory(mid=mid)
            break
        except Exception as e:
            print ('Error initing: {}'.format(e))
            k = input ('try again, or \'q\' to quit...')
            if k == 'q': break


    while True and k != 'q':
        if m.is_valid():
            print (m.get())
        else:
            print ('Memory not valid')
            try:
                m.reload()
            except Exception as e:
                print ('Error reloading: {}'.format(e))
        k = input ('Try to read again [\'q\' to quit this test]')

if has_zmes:
    i = input ('Test the Event Server? [y/N]').lower()
    if i=='y':
        ES_URL=None
        ES_USER=None
        ES_PASSWORD=None
        ALLOW_UNTRUSTED=True

        if not ES_URL: ES_URL = input ('Enter ES URL (example wss://foo:9000):')
        if not ES_USER: ES_USER = input ('Enter ES user (example admin):')
        if not ES_PASSWORD: ES_PASSWORD = getpass.getpass('Enter ES password:')

        es = ZMES({
            'url':ES_URL,
            'password': ES_PASSWORD,
            'user': ES_USER,
            'allow_untrusted': ALLOW_UNTRUSTED,
            'on_es_message': on_es_message,
            'on_es_error': on_es_error
                
        })
        # send a legit command
        
        print ("Sending a valid login")
        es.send({"event":"control","data":{"type":"filter","monlist":"1,2,5,6,7,8,9,10", "intlist":"0,0,0,0,0,0,0,0"}})
        print ("Sleeping for 3 seconds...")
        time.sleep(3)
        # send a bad command
        print ("Sending an invalid command")
        es.send ('Hi From ES')

        input ('press a key to proceed with the rest...')





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
    'from': '24 hours ago', # this will use localtimezone, use 'tz' for other timezones
    'object_only':False,
    'min_alarmed_frames': 1,
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

print ('Now trying to download an image from the first event')
print(cam_events.list())
if cam_events.list():
    e = cam_events.list()[0]
    print (e.name())
    e.download_image(dir='/tmp')
    e.download_video(dir='/tmp')
else:
    print ('No events found')
print ('Getting event summaries')
m = ms.find(name=cam_name)

# These will use server timezone
print ('Monitor {} has {} events {}'.format(m.name(), m.eventcount(options={'from':'1 hour ago','tz': zmapi.tz()}), '1 hour ago'))
print ('Monitor {} has {} events {}'.format(m.name(), m.eventcount(options={'from':'1 day ago','tz': zmapi.tz()}), '1 day ago'))

print ("--------| Getting ZM States |-----------")
states = zmapi.states()
for state in states.list():
    print ('State:{}[{}], active={}, details={}'.format(state.name(), state.id(), state.active(), state.definition()))

i = input ('Test Monitor State Change? [y/N]').lower()
if i=='y':
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

i = input ('Test ZM State Changes? [y/N]').lower()
if i=='y':
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


print ("--------| Configs Test |-----------")
try:
    conf = zmapi.configs()
    print (conf.find(name='ZM_AUTH_HASH_LOGINS'))
except Exception as e:
    print ('Error: {}'.format(str(e)))

zmlog.close()


