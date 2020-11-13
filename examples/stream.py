import pyzm
import pyzm.api as zmapi
import getpass
import traceback
import pyzm.ZMMemory as zmmemory
import time
#import pyzm.ml.object as  ObjectDetect
from pyzm.ml.detect_sequence import DetectSequence
from pyzm.helpers.Base import ConsoleLog
import pyzm.helpers.utils as utils
import sys

print ('Using pyzm version: {}'.format(pyzm.__version__))

logger = ConsoleLog()
logger.set_level(2)

if len(sys.argv) == 1:
    eid = input ('Enter event ID to analyze:')
else:
    eid = sys.argv[1]



'''
api_options = {
    'apiurl': 'https://demo.zoneminder.com/zm/api',
    'portalurl': 'https://demo.zoneminder.com/zm',
    'user': 'zmuser',
    'password': 'zmpass',
    'logger': logger, # use none if you don't want to log to ZM,
    #'disable_ssl_cert_check': True
}
'''

conf = utils.read_config('/etc/zm/secrets.ini')
api_options  = {
    'apiurl': utils.get(key='ZM_API_PORTAL', section='secrets', conf=conf),
    'portalurl':utils.get(key='ZM_PORTAL', section='secrets', conf=conf),
    'user': utils.get(key='ZM_USER', section='secrets', conf=conf),
    'password': utils.get(key='ZM_PASSWORD', section='secrets', conf=conf),
    'logger': logger, # use none if you don't want to log to ZM,
    #'disable_ssl_cert_check': True
}


zmapi = zmapi.ZMApi(options=api_options)

ml_options = {
    'sequence': 'object,face',

    'object': {
        'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
        'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
        'object_min_confidence': 0.3,
        'object_framework':'coral_edgetpu'
    },
    'face': {
        'face_detection_framework': 'dlib',
        'known_images_path': '/var/lib/zmeventnotification/known_faces',
        'face_model': 'cnn',
        'face_train_model': 'cnn',
        'face_recog_dist_threshold': 0.6,
        'face_num_jitters': 1,
        'face_upsample_times':1
    }
            
}

stream_options = {
        'frame_skip':3,
        #'start_frame': 10,
        #'max_frames':10,
        'strategy': 'most_unique',
        #'pattern': '(person|car|truck)',
        'api': zmapi,
        'download': False,
        'logger': logger,
        'frame_set': 'snapshot,alarm,23,25,29,30',
        #'resize': 800
}




#stream = '9130953'
#stream = '165242'
m = DetectSequence(options=ml_options)
#m = ObjectDetect.Object(options=ml_options)
b,l,c,f,_,a = m.detect_stream(stream=eid, options=stream_options)
print(a)
print (f'FRAME {f} with LABELS {l} {b} {c} ')
