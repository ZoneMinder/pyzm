from pyzm import __version__ as pyzmversion
import pyzm.api as zmapi
import getpass
import traceback
import pyzm.ZMMemory as zmmemory
import time
#import pyzm.ml.object as  ObjectDetect
from pyzm.ml.detect_sequence import DetectSequence
import pyzm.helpers.utils as utils
import sys
import pyzm.helpers.globals as g
import pyzm.ZMLog as log 


print ('Using pyzm version: {}'.format(pyzmversion))

#log.init(name='stream', override={'dump_console': True})
g.logger.set_level(5)


#time.sleep(1000)
mid = None

if len(sys.argv) == 1:
    eid = input ('Enter event ID to analyze:')
    mid = input ('Enter MID to use:')
else:
    eid = sys.argv[1]
    if len(sys.argv) == 2:
        print ('Event to analyze:{}'.format(eid))
        mid = input ('Enter MID to use:')
    else:
        mid = sys.argv[2]



'''
api_options = {
    'apiurl': 'https://demo.zoneminder.com/zm/api',
    'portalurl': 'https://demo.zoneminder.com/zm',
    'user': 'zmuser',
    'password': 'zmpass',
    #'disable_ssl_cert_check': True
}
'''

conf = utils.read_config('/etc/zm/secrets.ini')
api_options  = {
    'apiurl': utils.get(key='ZM_API_PORTAL', section='secrets', conf=conf),
    'portalurl':utils.get(key='ZM_PORTAL', section='secrets', conf=conf),
    'user': utils.get(key='ZM_USER', section='secrets', conf=conf),
    'password': utils.get(key='ZM_PASSWORD', section='secrets', conf=conf),
   # 'basic_auth_user': 'bob',
   # 'basic_auth_password': 'bobs password'
   #'disable_ssl_cert_check': True
}


zmapi = zmapi.ZMApi(options=api_options)
ml_options = {
    'general': {
        'model_sequence': 'object,face,alpr',
        'disable_locks': 'no'

    },
   
    'object': {
        'general':{
            'pattern':'.*',
            'same_model_sequence_strategy': 'most' # also 'most', 'most_unique's
        },
        'sequence': [{
            #First run on TPU
            'name': 'TPU for object detection', # descriptor (optional)
            'enabled': 'no', # skips TPU. Easy way to keep configs but not enable
            'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
            'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
            'object_min_confidence': 0.3,
            'object_framework':'coral_edgetpu'
        },
        {
            # YoloV4 on GPU if TPU fails (because sequence strategy is 'first')
             'name': 'GPU Yolov4 for object detection', # descriptor (optional)
            'enabled': 'no', # skips. Easy way to keep configs but not enable
            'object_config':'/var/lib/zmeventnotification/models/yolov4/yolov4.cfg',
            'object_weights':'/var/lib/zmeventnotification/models/yolov4/yolov4.weights',
            'object_labels': '/var/lib/zmeventnotification/models/yolov4/coco.names',
            'object_min_confidence': 0.3,
            'object_framework':'opencv',
            'object_processor': 'gpu',
            #'car_past_det_max_diff_area': '10%',
            #'match_past_detections': 'yes',
            #'car_max_detection_size': '13000',
            #'truck_max_detection_size': '13000',
            'image_path': '/var/lib/zmeventnotification/images',

            #'model_width': 512,
            #'model_height': 512
        }]
    },
    'face': {
        'general':{
            'pattern': '.*',
            'same_model_sequence_strategy': 'union'
        },
        'sequence': [{
            'name': 'DLIB face recognition',
            'enabled': 'yes',
            'face_detection_framework': 'dlib',
            'known_images_path': '/var/lib/zmeventnotification/known_faces',
            'face_model': 'cnn',
            'face_train_model': 'cnn',
            'face_recog_dist_threshold': 0.6,
            'face_num_jitters': 1,
            'face_upsample_times':1,
            'max_size': 800
        },
        {
            'name': 'TPU face detection',
            'enabled': 'yes',
            'face_detection_framework': 'tpu',
            'face_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite',

            'face_min_confidence': 0.3,
          
        }]
    },

    'alpr': {
         'general':{
            'same_model_sequence_strategy': 'first',
            'pre_existing_labels':['car', 'motorbike', 'bus', 'truck', 'boat'],

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
        #'frame_skip':2,
        #'start_frame': 21,
        #'max_frames':20,
        'strategy': 'most_models',
        #'strategy': 'first',
        'api': zmapi,
        'download': False,
        'frame_set': 'snapshot,alarm',
        'resize': 800,
        'save_frames': False,
        'save_analyzed_frames': False,
        'save_frames_dir': '/tmp',
        'contig_frames_before_error': 5,
        'max_attempts': 3,
        'sleep_between_attempts': 4,
        'disable_ssl_cert_check': True,
        'mid':mid
}


#input ('Enter...')
m = DetectSequence(options=ml_options)
#m = ObjectDetect.Object(options=ml_options)
matched_data,all_data = m.detect_stream(stream=eid, options=stream_options)
print('ALL FRAMES: {}\n\n'.format(all_data))
print ('SELECTED FRAME: {}, SIZE: {}  LABELS: {} BOXES:{} CONFIDENCES:{}'.format(matched_data['frame_id'],matched_data['image_dimensions'],matched_data['labels'],matched_data['boxes'],matched_data['confidences']))

