import pyzm
import pyzm.api as zmapi
import getpass
import traceback
import pyzm.ZMMemory as zmmemory
import time
import pyzm.ml.object as  ObjectDetect
from pyzm.helpers.Base import ConsoleLog
import pyzm.helpers.utils as utils

print ('Using pyzm version: {}'.format(pyzm.__version__))

logger = ConsoleLog()
logger.set_level(2)

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
'''

zmapi = zmapi.ZMApi(options=api_options)

ml_options = {
            'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
            'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
            'object_min_confidence': 0.3,
            'object_framework':'coral_edgetpu'
        }
'''
ml_options = {
            'object_weights':'/var/lib/zmeventnotification/models/yolov4/yolov4.weights',
            'object_labels': '/var/lib/zmeventnotification/models/yolov4/coco.names',
            'object_config': '/var/lib/zmeventnotification/models/yolov4/yolov4.cfg',
            'object_min_confidence': 0.3,
            'object_framework':'opencv',
            'object_processor': 'gpu'
        }
'''


stream_options = {
        'frame_skip':3,
        'start_frame': 10,
        'max_frames':10,
        'strategy': 'most_unique',
        #'pattern': '(person|car|truck)',
        'api': zmapi,
        'download': False,
        'logger': logger,
        #'frame_set': 'snapshot,alarm,2,3,5,7,9,20,30,40,100,300'
}




stream = '9130953'
m = ObjectDetect.Object(options=ml_options)
b,l,c,f,_,a = m.detect_stream(stream=stream, options=stream_options)
print(a)
print (f'FRAME {f} with LABELS {l}')

