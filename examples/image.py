from pyzm import __version__ as pyzmversion
import traceback
import time
#import pyzm.ml.object as  ObjectDetect
from pyzm.ml.detect_sequence import DetectSequence
import pyzm.helpers.utils as utils
import sys
import pyzm.helpers.globals as g


g.logger.Info ('Using pyzm version: {}'.format(pyzmversion))



#time.sleep(1000)
mid = None

if len(sys.argv) == 1:
    stream = input ('Enter filename to analyze: ')
else:
    stream = sys.argv[1]

conf = utils.read_config('/etc/zm/secrets.yml')

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
            'enabled': 'yes', # skips TPU. Easy way to keep configs but not enable
            'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
            'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
            'object_min_confidence': 0.3,
            'object_framework':'coral_edgetpu'
        },
        {
            # YoloV4 on GPU if TPU fails (because sequence strategy is 'first')
             'name': 'GPU Yolov4 for object detection', # descriptor (optional)
            'enabled': 'yes', # skips. Easy way to keep configs but not enable
            'object_config':'/var/lib/zmeventnotification/models/yolov4/yolov4.cfg',
            'object_weights':'/var/lib/zmeventnotification/models/yolov4/yolov4.weights',
            'object_labels': '/var/lib/zmeventnotification/models/yolov4/coco.names',
            'object_min_confidence': 0.3,
            'object_framework':'opencv',
            'object_processor': 'cpu',
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
            #'platerec_payload': {
                #'regions':['us'],
                #'camera_id':12,
            #},
            #'platerec_config': {
            #    'region':'strict',
            #    'mode': 'fast'
            #}
         }]
    }
} # ml_options

stream_options = {
        'strategy': 'most_models',
}


#input ('Enter...')
m = DetectSequence(options=ml_options)
#m = ObjectDetect.Object(options=ml_options)
matched_data,all_data = m.detect_stream(stream=stream, options=stream_options)
print('ALL FRAMES: {}\n\n'.format(all_data))
print ('SELECTED FRAME: {}, SIZE: {}  LABELS: {} BOXES:{} CONFIDENCES:{}'.format(matched_data['frame_id'],matched_data['image_dimensions'],matched_data['labels'],matched_data['boxes'],matched_data['confidences']))

