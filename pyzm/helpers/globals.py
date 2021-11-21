from datetime import datetime

from yaml import safe_load
from typing import Optional

#MLAPI
MAX_FILE_SIZE_MB: int = 5
ALLOWED_EXTENSIONS: set = {'.png', '.jpg', '.gif', '.mp4'}
ACCESS_TOKEN_EXPIRES: int = 60 * 60  # 1 hr
#ZMES
animation_seconds: Optional[datetime] = None  # Time it takes to create an animation
# COMMON
logger: Optional = None  # global logger, starts with a buffer that is displayed once ZMLog is initialized
config: Optional[dict] = {}  # object that will hold config values from objectconfig.yml
api: Optional = None  # global zmapi
mid: Optional[int] = None  # global Monitor ID
eid: Optional[str] = None  # global Event ID or the name of the input file/video

api_event_response: Optional[dict] = None  # return from requesting event data
event_tot_frames: Optional[int] = None  # Total frame buffer length for current event / video /image

Frame: Optional[list] = None  # Hold the events 'Frame' data structure (length = length of frame buffer)
Monitor: Optional[dict] = None  # Hold the events 'Monitor' data structure
Event: Optional[dict] = None  # Hold the events 'Event' data structure


#  Default Configuration Values
# Todo: Move to config file and integrate with ENV variables like pyzm is
zmes_default: dict = safe_load('''
custom_push: no
custom_push_script: ''
force_mpd: no
same_model_high_conf: no
skip_mons: 
force_live: no
sanitize_logs: no
sanitize_str: <sanitized>
show_models: no
save_image_train: no
save_image_train_dir: '/var/lib/zmeventnotification/images'
force_debug: no
frame_set: snapshot,alarm,snapshot
cpu_max_processes: 2
gpu_max_processes: 2
tpu_max_processes: 2
cpu_max_lock_wait: 120
gpu_max_lock_wait: 120
tpu_max_lock_wait: 120
pyzm_overrides: '{ "log_level_debug" : 5 }'
secrets: ''
user: ''
password: ''
basic_user: ''
basic_password: ''
portal: ''
api_portal: ''
image_path: '/var/lib/zmeventnotification/images'
allow_self_signed: yes
match_past_detections: no
past_det_max_diff_area: '5%'
max_detection_size: ''
contained_area: 1px
model_sequence: 'object,face,alpr'
base_data_path: '/var/lib/zmeventnotification'
resize: no
picture_timestamp:
  enabled: no
  date format: '%Y-%m-%d %H:%M:%S'
  monitor id: yes
  text color: (255,255,255)
  background: yes
  bg color: (0,0,0)

delete_after_analyze: yes
write_debug_image: no
write_image_to_zm: yes
show_percent: yes
draw_poly_zone: yes
contained_area: 1px
poly_color: (0,0,255)
poly_thickness: 2
import_zm_zones: no
only_triggered_zm_zones: no
show_filtered_detections: no
show_conf_filtered: no

hass_enabled: no
hass_server: ''
hass_token: ''

hass_people: {}
hass_notify: ''
hass_cooldown: ''

push_enable: no
push_force: no
push_token: ''
push_key: ''

push_url: no
push_user: ''
push_pass: ''

push_errors: no
push_err_token: ''
push_err_key: ''
push_err_device: ''

push_jpg: ''
push_jpg_key: ''
push_gif: ''
push_gif_key: ''

push_debug_device: '' 
push_cooldown: ''

mqtt_enable: no
mqtt_force: no
mqtt_topic: ''
mqtt_broker: ''
mqtt_port: '' 
mqtt_user: ''
mqtt_pass: ''
mqtt_tls_allow_self_signed: no
mqtt_tls_insecure: no
tls_ca: ''
tls_cert: ''
tls_key: ''

create_animation: no
animation_timestamp:
  enabled: no
  date format: '%Y-%m-%d %H:%M:%S'
  monitor id: yes
  text color: (255,255,255)
  background: yes
  bg color: (0,0,0)
animation_types: 'gif,mp4'
fast_gif: no
animation_width: 640
animation_retry_sleep: 3
animation_max_tries: 8

ml_fallback_local: no
ml_enable: no
ml_routes: []

object_detection_pattern: '(person|car|motorbike|bus|truck|boat|dog|cat)'
object_min_confidence: 0.6
tpu_object_labels: '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names'
tpu_object_framework: coral_edgetpu
tpu_object_processor: tpu
tpu_min_confidence: 0.6

yolo4_object_weights: '/var/lib/zmeventnotification/models/yolov4/yolov4.weights'
yolo4_object_labels: '/var/lib/zmeventnotification/models/yolov4/coco.names'
yolo4_object_config: '/var/lib/zmeventnotification/models/yolov4/yolov4.cfg'
yolo4_object_framework: opencv
yolo4_object_processor: gpu
fp16_target: no

yolo3_object_weights: '/var/lib/zmeventnotification/models/yolov3/yolov3.weights'
yolo3_object_labels: '/var/lib/zmeventnotification/models/yolov3/coco.names'
yolo3_object_config: '/var/lib/zmeventnotification/models/yolov3/yolov3.cfg'
yolo3_object_framework: opencv
yolo3_object_processor: gpu

tinyyolo_object_config: '/var/lib/zmeventnotification/models/tinyyolov4/yolov4-tiny.cfg'
tinyyolo_object_weights: '/var/lib/zmeventnotification/models/tinyyolov4/yolov4-tiny.weights'
tinyyolo_object_labels: '/var/lib/zmeventnotification/models/tinyyolov4/coco.names'
tinyyolo_object_framework: opencv
tinyyolo_object_processor: gpu

face_detection_pattern: .*
known_images_path: '/var/lib/zmeventnotification/known_faces'
unknown_images_path: '/var/lib/zmeventnotification/unknown_faces'
save_unknown_faces: no
save_unknown_faces_leeway_pixels: 100
face_detection_framework: dlib
face_dlib_processor: gpu
face_num_jitters: 1
face_model: cnn
face_upsample_times: 1
face_recog_dist_threshold: 0.6
face_train_model: cnn
unknown_face_name: Unknown

alpr_detection_pattern: .*
alpr_api_type: ''
alpr_service: ''
alpr_url: ''
alpr_key: ''
platerec_stats: no
platerec_regions: []
platerec_min_dscore: 0.1
platerec_min_score: 0.2

openalpr_recognize_vehicle: 0
openalpr_country: '' 
openalpr_state: ''
openalpr_min_confidence: 0.3

openalpr_cmdline_binary: alpr
openalpr_cmdline_params: -j -d
openalpr_cmdline_min_confidence: 0.3

smart_fs_thresh :  5
disable_locks :  no
frame_strategy :  first
same_model_sequence_strategy :  most

stream_sequence :
  frame_strategy: '{{frame_strategy}}'
  frame_set: '{{frame_set}}'
  contig_frames_before_error: 2
  max_attempts: 3
  sleep_between_attempts: 2.23
  sleep_between_frames: 0
  sleep_between_snapshots: 1.5
  smart_fs_thresh: '5'
  resize: '{{resize}}'
  model_height:  
  model_width: 
  tpu_model_height: 
  tpu_model_width: 

ml_sequence:
  general:
    model_sequence: '{{model_sequence}}'
    disable_locks: no

  object:
      general:
        object_detection_pattern: '(person|dog|cat|car|truck)'
        same_model_sequence_strategy: '{{same_model_sequence_strategy}}'
        contained_area: 1px

      sequence:
        - name: 'Yolo v4'
          enabled: 'yes'
          object_config: '{{yolo4_object_config}}'
          object_weights: '{{yolo4_object_weights}}'
          object_labels: '{{yolo4_object_labels}}'
          object_min_confidence: '{{object_min_confidence}}'
          object_framework: '{{yolo4_object_framework}}'
          object_processor: '{{yolo4_object_processor}}'
          gpu_max_processes: '{{gpu_max_processes}}'
          gpu_max_lock_wait: '{{gpu_max_lock_wait}}'
          cpu_max_processes: '{{cpu_max_processes}}'
          cpu_max_lock_wait: '{{cpu_max_lock_wait}}'
          fp16_target: '{{fp16_target}}'  # only applies to GPU, default is 'no'

  alpr:
    general:
      same_model_sequence_strategy: 'first'
      alpr_detection_pattern: '{{alpr_detection_pattern}}'

    sequence: []

  face:
    general:
      face_detection_pattern: '{{face_detection_pattern}}'
      same_model_sequence_strategy: 'union'

      sequence: []
''')

mlapi_default: dict = safe_load("""
host: 0.0.0.0
processes: 1
port: 5000
wsgi_server: flask

model_sequence: 'object'
same_model_high_conf: no
sanitize_logs: no
sanitize_str: <sanitized>
log_user:
log_group:
log_name: 'zm_mlapi'
log_path: './logs'
base_data_path: '.'
match_past_detections: no
past_det_max_diff_area: '5%'


zmes_keys: {}
frame_set: snapshot,alarm,snapshot
force_mpd: no
secrets: ./mlapisecrets.yml
auth_enabled: yes
import_zm_zones: no
only_triggered_zm_zones: no
cpu_max_processes: 2
cpu_max_lock_wait: 120
gpu_max_processes: 2
gpu_max_lock_wait: 120
tpu_max_processes: 2
tpu_max_lock_wait: 120

image_path: './images'
db_path: './db'
wait: 0.0
mlapi_secret_key: 'ChangeMe this is for creating the auth JWT for users to connect'
max_detection_size: 90%
contained_area: '1px'
detection_sequence: object
pyzm_overrides: {}

smart_fs_thresh: 5
disable_locks: no
object_framework: opencv
object_detection_pattern: (person|dog|car|truck)
object_min_confidence: 0.6
fp16_target: no
show_models: no

face_detection_pattern: .*
face_detection_framework: dlib
face_recognition_framework: dlib
face_num_jitters: 0
face_upsample_times: 0
face_model: cnn
face_train_model: cnn
face_recog_dist_threshold: 0.6
face_recog_knn_algo: ball_tree
face_dlib_processor: gpu
unknown_face_name: Unknown_Face
save_unknown_faces: no
save_unknown_faces_leeway_pixels: 100
alpr_detection_pattern: .*
alpr_api_type: cloud
alpr_service: 
alpr_url: 
alpr_key: 
platerec_stats: no
platerec_regions: [ ]
platerec_min_dscore: 0.1
platerec_min_score: 0.2

openalpr_recognize_vehicle: 1
openalpr_country: us
openalpr_state: ca
openalpr_min_confidence: 0.3

openalpr_cmdline_binary: alpr
openalpr_cmdline_params: -j -d
openalpr_cmdline_min_confidence: 0.3

stream_sequence:
  frame_strategy: 'most'
  frame_set: snapshot,alarm,snapshot
  contig_frames_before_error: 2
  max_attempts: 4
  sleep_between_attempts: 2
  sleep_between_frames: 0.3
  sleep_between_snapshots: 2
  smart_fs_thresh: 5
  resize: no
  model_height: 416 
  model_width: 416
  tpu_model_height: 320
  tpu_model_width: 320

ml_sequence:
  general:
    model_sequence: object
    disable_locks: no

  object:
      general:
        object_detection_pattern: '(person|dog|cat|car|truck)'
        same_model_sequence_strategy: most
        contained_area: 1px

      sequence:
        - name: 'BUILT IN YOLO V4'
          enabled: 'yes'
          object_config: '{{yolo4_object_config}}'
          object_weights: '{{yolo4_object_weights}}'
          object_labels: '{{yolo4_object_labels}}'
          object_min_confidence: '{{object_min_confidence}}'
          object_framework: '{{yolo4_object_framework}}'
          object_processor: 'gpu'
          gpu_max_processes: '{{gpu_max_processes}}'
          gpu_max_lock_wait: '{{gpu_max_lock_wait}}'
          cpu_max_processes: '{{cpu_max_processes}}'
          cpu_max_lock_wait: '{{cpu_max_lock_wait}}'
          fp16_target: 'no'  # only applies to GPU, default is FP32
          show_models: 'no'  # at current moment this is a global setting turned on by just setting it to : yes
  alpr:
    general:
      same_model_sequence_strategy: 'first'
      alpr_detection_pattern: '.*'

    sequence: []

  face:
    general:
      face_detection_pattern: '.*'
        # combine results below
      same_model_sequence_strategy: 'union'

      sequence: []
""")
