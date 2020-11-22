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
    'general': {
        'model_sequence': 'object,face,alpr',
    
    },
   
    'object': {
        'general':{
            'same_model_sequence_strategy': 'first' # also 'most', 'most_unique's
        },
        'sequence': [{
            #First run on TPU
            'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
            'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
            'object_min_confidence': 0.3,
            'object_framework':'coral_edgetpu'
        },
        {
            # YoloV4 on GPU if TPU fails (because sequence strategy is 'first')
            'object_config':'/var/lib/zmeventnotification/models/yolov4/yolov4.cfg',
            'object_weights':'/var/lib/zmeventnotification/models/yolov4/yolov4.weights',
            'object_labels': '/var/lib/zmeventnotification/models/yolov4/coco.names',
            'object_min_confidence': 0.3,
            'object_framework':'opencv',
            'object_processor': 'gpu'
        }]
    },
    'face': {
        'general':{
            'same_model_sequence_strategy': 'first'
        },
        'sequence': [{
            'face_detection_framework': 'dlib',
            'known_images_path': '/var/lib/zmeventnotification/known_faces',
            'face_model': 'cnn',
            'face_train_model': 'cnn',
            'face_recog_dist_threshold': 0.6,
            'face_num_jitters': 1,
            'face_upsample_times':1
        }]
    },

    'alpr': {
         'general':{
            'same_model_sequence_strategy': 'first'
        },
         'sequence': [{
            'alpr_api_type': 'cloud',
            'alpr_service': 'plate_recognizer',
            'alpr_key': utils.get(key='PLATEREC_ALPR_KEY', section='secrets', conf=conf),
            'platrec_stats': 'no',
            'platerec_min_dscore': 0.1,
            'platerec_min_score': 0.2,
         }]
    }
} # ml_options

stream_options = {
        'frame_skip':2,
        'start_frame': 21,
        'max_frames':10,
        'strategy': 'most_unique',
        #'pattern': '(person|car|truck)',
        'api': zmapi,
        'download': False,
        'logger': logger,
        'frame_set': 'snapshot,alarm'
        #'resize': 800
}




#stream = '9130953'
#stream = '165242'
m = DetectSequence(options=ml_options)
#m = ObjectDetect.Object(options=ml_options)
b,l,c,f,_,a = m.detect_stream(stream=eid, options=stream_options)
print(f'ALL FRAMES: {a}\n\n')
print (f'SELECTED FRAME {f} with LABELS {l} {b} {c} ')

