import pyzm
import pyzm.api as zmapi
import getpass
import traceback
import pyzm.ZMMemory as zmmemory
import time
import pyzm.ml.object as  ObjectDetect
from pyzm.helpers.Base import SimpleLog

print ('Using pyzm version: {}'.format(pyzm.__version__))

logger = SimpleLog()

api_options = {
    'apiurl': 'https://demo.zoneminder.com/zm/api',
    'portalurl': 'https://demo.zoneminder.com/zm',
    'user': 'zmuser',
    'password': 'zmpass',
    'logger': logger, # use none if you don't want to log to ZM,
    #'disable_ssl_cert_check': True
}

zmapi = zmapi.ZMApi(options=api_options)


ml_options = {
            'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
            'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
            'object_min_confidence': 0.3,
            'object_framework':'coral_edgetpu'
        }



stream_options = {
        'frame_skip':3,
        'start_frame': 1,
        'max_frames':10,
        'strategy': 'most_unique',
        'pattern': '.*',
        'api': zmapi,
        'download': False,
        'logger': logger
}




stream = '9130943'
m = ObjectDetect.Object(options=ml_options)
b,l,c,f,_,a = m.detect_stream(stream=stream, options=stream_options)
print(a)

