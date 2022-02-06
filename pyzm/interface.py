from ast import literal_eval
from dataclasses import dataclass, field
import time
from hashlib import new
from pathlib import Path
from typing import Optional, Union
from re import compile

from pyzm.ZMLog import str2bool
from yaml import safe_load

SECRETS_REGEX = r"^\b|\s*(\w.*):\s*\"?|\'?({\[\s*(\w.*)\s*\]})\"?|\'?"
SUBVAR_REGEX = r"^\b|\s*(\w.*):\s*\"?|\'?({{\s*(\w.*)\s*}})\"?|\'?"
ZMES_DEFAULT_CONFIG: dict = safe_load(
    """
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

    push_emergency: no
    #push_emerg_mons: ''
    push_emerg_expire: 3600
    push_emerg_retry: 60
    push_emerg_time_start: '00:00'
    push_emerg_time_end: '23:59'
    push_emerg_force: no
    
    goti_enable: no
    goti_host: ''
    goti_token: ''

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

    ml_sequence:
      general:
        model_sequence: '{{model_sequence}}'
        disable_locks: no

      object:
        general:
          object_detection_pattern: '(person|dog|cat|car|truck)'
          same_model_sequence_strategy: '{{same_model_sequence_strategy}}'
          contained_area: '1px'
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
    """
)
MLAPI_DEFAULT_CONFIG: dict = safe_load(
    """

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
"""
)


class Singleton(type):
    """Python implementation of the Singleton pattern, to be used as a metaclass."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass()
class GlobalConfig(metaclass=Singleton):
    """Customized dataclass to hold global objects that ZMES utilizes"""

    DEFAULT_CONFIG: dict = field(default_factory=dict)  # hardcoded config values

    eid: Optional[Union[str, int]] = None  # global Event ID or the name of the input file/video
    mid: Optional[int] = None  # global Monitor ID
    logger = None  # global logger, starts with a buffer that is displayed once ZMLog is initialized
    api = None  # global ZMApi -> todo: fix circular imports
    config: dict = field(default_factory=dict)  # object that will hold active config values

    animation_seconds: Optional[time.perf_counter] = None  # how long it takes for an animation to be created
    api_event_response: Optional[dict] = None  # return from requesting event data
    event_tot_frames: Optional[int] = None  # Total frame buffer length for current event / video /image
    Frame: Optional[list] = None  # Hold the events 'Frame' data structure (length = length of frame buffer)
    Monitor: Optional[dict] = None  # Hold the events 'Monitor' data structure
    Event: Optional[dict] = None  # Hold the events 'Event' data structure

    extras: dict = field(default_factory=dict)  # Extras


g: GlobalConfig


class ZMESConfig:
    """
    :param dict hardcoded_config: Configuration defaults that are hardcoded, these are added to the config before substitution.
        :param str config_type: The type of configuration file - zmes, mlapi or other.
        :param str config_file_path: Absolute path to the YAML configuration or secrets file.
        :param bool no_auto_parse: If set to True it will not call parse() at the end of init, meaning the user will need to call it themselves.

    """

    hardcoded_config: Optional[dict]
    config_type: str
    config_hash: str
    secrets_hash: str
    secrets_pool: dict
    config_file_path: str
    secrets_file_path: str
    detection_patterns: dict
    polygons: dict
    monitors: Optional[dict]
    built_monitors: dict
    built_per_mon_configs: dict
    raw_config: dict
    config: dict

    def __init__(
        self,
        config_file_path: str,
        hardcoded_config: Optional[dict] = None,
        config_type: Optional[str] = None,
        no_auto_parse: bool = False,
    ):
        """A class to parse YAML syntax config and secrets files into a structure for ZMES or MLAPI.

        This will substitute {[secrets]} and {{substitution variables}}. It also includes a file hashing method.
        There is a method to build an overridden configuration based on the per-monitor 'monitors' section

        :param dict hardcoded_config: Configuration defaults that are hardcoded, these are added to the config before substitution.
        :param str config_type: The type of configuration file - zmes, mlapi or other.
        :param str config_file_path: Absolute path to the YAML configuration or secrets file.
        :param bool no_auto_parse: If set to True it will not call parse() at the end of init, meaning the user will need to call it themselves.
        """
        global g
        g = GlobalConfig()

        lp: str = "conf:init:"
        hcc = self.hardcoded_config = hardcoded_config
        self.config_type = config_type or "other"
        self.config_hash = ""
        self.secrets_hash = ""
        self.secrets_pool = {}

        self.config_file_path = config_file_path
        self.secrets_file_path = ""

        # Custom detection patterns per monitor
        self.detection_patterns = {}
        # Polygon coords per monitor
        self.polygons = {}
        # Will hold the 'monitors' section, it will be parsed to substitute secrets and sub vars
        self.monitors = None
        # Will hold the processed 'monitors' section and a global config after parsing each per monitor config
        self.built_monitors = {}
        # Holds the g.config that is built using per-monitor section
        self.built_per_mon_configs = {}
        # The yaml config, will hold the config as it was read from file
        self.raw_config = {}
        # Active build config - will be g.config
        self.config = {}

        if hcc and not config_type:
            g.logger.debug(
                f"{lp} config_type was not supplied, a hardcoded config was, ascertaining config file type by "
                f"searching hardcoded keys..."
            )
            if hcc.get("mlapi_secret_key") is not None:
                self.config_type = "mlapi"
            elif hcc.get("create_animation") is not None:
                self.config_type = "zmes"
            else:
                self.config_type = "other"
        elif not hcc and not config_type:
            g.logger.error(f"{lp} neither config_type nor hardcoded config was supplied")
            raise SyntaxError("A 'config_type' or a 'hardcoded_config' need to be supplied")

        g.logger.debug(f"{lp} config type is '{self.config_type}'")
        if not no_auto_parse:
            self.parse()

    def hash(
        self,
        input_file: Optional[Union[Path, str]] = None,
        input_hash: Optional[str] = None,
        comparative_hash: Optional[str] = None,
        read_chunk_size: int = 65536,
        algorithm: str = "sha256",
    ) -> tuple[str, Optional[bool]]:
        """Hash a file using hashlib library.

        If an ``input_file`` is passed, that file will be read and hashed into a string. If a ``input_hash`` is supplied

        Default algorithm is SHA-256, see hashlib

        :param input_hash: (Optional) Instead of hashing a file, supply the hash
        :param str|Path input_file: Pre-instantiated Pathlib.Path object or a str with an absolute path
        :param int comparative_hash: Hash to compare to
        :param int read_chunk_size: Maximum number of bytes to be read from the file
         at once. Default is 65536 bytes or 64KB
        :param str algorithm: The hash algorithm name to use. For example, 'md5',
         'sha256', 'sha512' and so on. Default is 'sha256'. Refer to
         hashlib.algorithms_available for available algorithms
        :return: a tuple with the calculated hash and if a comparative_hash was supplied, its result
        :rtype tuple:
        """

        compare_ret: Optional[bool] = None
        cached_hash: Optional[str] = None
        lp: str = "conf:hash:"
        if input_file is None and input_hash is None:
            g.logger.error(f"{lp} no inputs at all?!?!")
            raise SyntaxError("There must be something to process")
        elif input_file is not None and input_hash is not None:
            g.logger.warning(f"{lp} an input_file and input_hash were supplied, file takes precedence")
            input_hash = None
        elif input_hash:
            cached_hash = input_hash

        if input_hash is None and input_file:
            # input_file provided
            if isinstance(input_file, str):
                g.logger.debug(f"{lp} string containing path provided, converting to pathlib.Path object")
                input_file = Path(input_file)
            elif not isinstance(input_file, Path):
                g.logger.error(f"{lp} the 'input_file' param is not a string or pathlib.Path object!")
                raise TypeError("'input_file' argument must be a string with absolute path or a pathlib.Path object!")

            if input_file.exists() and input_file.is_file():
                checksum = new(algorithm)  # Raises appropriate exceptions.
                try:
                    with input_file.open("rb") as f:
                        for chunk in iter(lambda: f.read(read_chunk_size), b""):
                            checksum.update(chunk)
                except Exception as exc:
                    g.logger.error(f"{lp} ERROR while computing {algorithm} hash of '{input_file.name}'")
                    raise exc
                else:
                    cached_hash = checksum.hexdigest()
                    g.logger.debug(f"{lp} the {algorithm} hex digest for file '{input_file.name}' -> {cached_hash}")
            else:
                g.logger.error(f"{lp} 'input_file' {input_file} is invalid (permissions, does not exist, etc.)")
                raise FileNotFoundError

        if comparative_hash is not None:
            if comparative_hash:
                g.logger.debug(f"{lp} comparing hashes - PROVIDED: {cached_hash} -- COMPARING TO: {comparative_hash}")
            else:
                g.logger.debug(f"{lp} the comparative_hash provided is empty!")
            compare_ret = cached_hash == comparative_hash

        return cached_hash, compare_ret

    def parse(self, force_config_hash: bool = False, force_secrets_hash: bool = False):
        """Begin parsing the configuration file.
        There is logic to hash the mlapi config file

        :param bool force_config_hash: Force hashing of the supplied config file. 'mlapi' config_type is hashed by default.
        """
        # Validate the config file
        lp: str = "conf:parse:"
        cfn: str = self.config_file_path
        rc: dict = self.raw_config
        if Path(cfn).exists() and Path(cfn).is_file():
            g.logger.debug(f"{lp} supplied config file '{cfn}' is valid")
            if self.config_type == "mlapi" or force_config_hash:
                # Hash the config file and cache the result
                self.config_hash, _ = self.hash(Path(cfn))
            try:
                # Read the supplied config file into a python dict using pyyaml safe_load
                with open(cfn, "r") as stream:
                    rc = safe_load(stream)
            except TypeError as e:
                g.logger.error(f"{lp} the supplied config file is not valid YAML -> '{cfn}'")
                raise e
            except Exception as exc:
                g.logger.error(f"{lp} error trying to load YAML in config file -> '{cfn}'")
                g.logger.debug(exc)
            else:
                tmp_yaml_cfg = dict(rc)
                # Flatten the sections to get all the keys to the 'base' level
                if str2bool(rc.get("SECTIONS")) or str2bool(rc.get("MLAPI")) or str2bool(rc.get("ZMES")):
                    sections = tmp_yaml_cfg.keys()
                    g.logger.debug(f"{lp} '{self.config_type}' config file has 'sections' enabled, flattening...")
                    for section in sections:
                        if section == "monitors":
                            self.config[section] = tmp_yaml_cfg[section]
                        elif isinstance(tmp_yaml_cfg[section], dict):
                            for k, v in tmp_yaml_cfg[section].items():
                                if k not in self.config:
                                    self.config[k] = v
                else:
                    self.config = tmp_yaml_cfg

                # Grab from rc as we want it straight from the freshly parsed file
                self.monitors = rc.get("monitors", {})
                g.logger.debug(f"{lp} YAML configuration parsed (no secrets or substitution vars replaced)")
                if hcc := self.hardcoded_config:  # WALRUS BOI
                    def_keys_added: list = []
                    # If the config does not contain a default key, add it
                    for default_key, default_value in hcc.items():
                        if default_key not in self.config:
                            def_keys_added.append(default_key)
                            self.config[default_key] = default_value
                    if def_keys_added:
                        g.logger.debug(
                            f"{lp} {len(def_keys_added)} hardcoded configuration option"
                            f"{'' if len(def_keys_added) == 1 else 's'} added to the supplied "
                            f"config -> {def_keys_added}"
                        )

                # Automated logic to build per-monitor overridden configurations
                if g.mid and self.config_type == "zmes":
                    g.logger.debug(
                        f"{lp}mid {g.mid}: '{self.config_type}' copying current supplied build config to monitor "
                        f"{g.mid} overridden config as a template"
                    )
                    self.built_per_mon_configs[g.mid] = dict(self.config)
                elif self.config_type == "mlapi":
                    if self.monitors is not None and isinstance(self.monitors, dict):
                        for mon_id in self.monitors.keys():
                            g.logger.debug(
                                f"{lp}mid {mon_id}: '{self.config_type}' copying current supplied build config "
                                f"to monitor {mon_id} overridden config as a template"
                            )
                            self.built_per_mon_configs[mon_id] = dict(self.config)
                else:
                    g.logger.warning(
                        f"{lp} Unknown situation! {self.config_type} config file type, how to parse per-monitor stuff?"
                    )

                # Substitute {[secrets]} then {{sub vars}}
                sfn = self.secrets_file_path = self.config.get("secrets", "")
                # todo: {{sub var}} in secrets name support to allow for "{{base_data_path}}/secrets.yml"

                try:
                    self.config = self._parse_secrets()
                    self.config = self._parse_vars()
                except Exception as e:
                    g.logger.error(f"{lp} '{self.config_type}' substitution error! -> {e}")
                    g.logger.log_close()
                    raise e

                if self.config_type == "mlapi":
                    if sfn:
                        self.secrets_hash, _ = self.hash(Path(sfn))
                    if self.monitors and isinstance(self.monitors, dict):
                        g.logger.debug(f"{lp} building per-monitor overrode configs")
                        for mon_id, per_mon_config in self.monitors.items():
                            if per_mon_config is not None and self.built_per_mon_configs.get(mon_id) is not None:
                                self.built_per_mon_configs[mon_id] = self.override_monitor(
                                    mon_id, self.built_per_mon_configs[mon_id]
                                )
                    # Ensure setting resize in mlapi config file will not have any effect
                    # ZMES controls 'resize' option
                    if self.config.get("resize") is not None:
                        self.config.pop("resize")
                elif self.config_type == "zmes":
                    if g.mid in self.monitors:
                        g.logger.debug(f"{lp} building per-monitor overrode config for monitor {g.mid}")
                        self.built_per_mon_configs[g.mid] = self.override_monitor(
                            g.mid, self.built_per_mon_configs[g.mid]
                        )
                else:
                    g.logger.warning(
                        f"{lp} unknown situation! '{self.config_type}' config file type, how to parse "
                        f"per-monitor stuff?"
                    )
        else:
            g.logger.error(f"{lp} config file {cfn} not found. Check permissions?")

    def _parse_secrets(
        self,
        config: Optional[dict] = None,
        secrets_path: Optional[str] = None,
        secrets_pool: Optional[dict] = None,
    ) -> dict:
        """
        :param dict config: (Optional) The configuration to search and replace {[secrets]} in
        :param str secrets_path: (Optional) The secrets file to parse into a dict
        :param dict secrets_pool: (Optional) The secrets pool to use for substitution
        :return: New dict with replaced secrets
        """
        lp = "conf:secrets:"
        sfn: Optional[str] = None
        return_config: Union[dict, str] = {}
        if config is None and self.config:
            config = self.config
        else:
            g.logger.debug(f"{lp} a config was supplied to search and substitute")

        if secrets_path is None and secrets_pool is None and self.secrets_file_path:
            sfn = self.secrets_file_path
        elif secrets_path is None and secrets_pool is None and not self.secrets_file_path:
            g.logger.warning(
                f"{lp} there is no secrets file or a secrets pool to substitute from, "
                f"skipping parsing {{[secrets]}}"
            )
        elif secrets_path is not None:
            sfn = secrets_path
            g.logger.debug(f"{lp} a 'secrets_path' was supplied to parse a YAML file")

        if sfn and Path(sfn).exists() and Path(sfn).is_file():
            g.logger.debug(f"{lp} the supplied secrets file '{sfn}' exists")
            try:
                with Path(sfn).open("r") as stream:
                    self.secrets_pool = secrets_pool = safe_load(stream)
            except Exception as exc:
                g.logger.error(f"{lp} an exception occurred while trying to load YAML from '{sfn}'")
                raise exc
            else:
                g.logger.debug(f"{lp} '{sfn}' was parsed from YAML to a dictionary")
        elif sfn and not Path(sfn).exists():
            g.logger.error(f"{lp} the configured secrets file does not exist -> '{sfn}'")
            raise FileNotFoundError
        elif sfn and not Path(sfn).is_file():
            g.logger.error(f"{lp} the configured secrets file exists but it is not a file! -> '{sfn}'")
            raise TypeError("the configured secrets file exists but it is not a file!")

        if secrets_pool:
            secrets_replaced: list = []
            secrets_not_replaced: list = []
            g.logger.debug(f"{lp} starting '{{[secrets]}}' search and substitution")
            return_config = str(config)
            found_secrets = set(compile(r"\{\[(\w*)\]\}").findall(return_config))
            for secret_ in found_secrets:
                if secret_ in secrets_pool and secrets_pool[secret_] is not None:
                    # the extracted secret has a key: value in the secrets_pool
                    secrets_replaced.append(secret_)
                    # For some reason a regular string would produce weird results, this works
                    pattern: str = r"".join([r"\{\[", r"{}".format(secret_), r"\]\}"])
                    return_config = compile(pattern=pattern).sub(secrets_pool[secret_], return_config)
                else:
                    secrets_not_replaced.append(secret_)
            if secrets_replaced:
                g.logger.debug(
                    f"{lp} successfully replaced {len(secrets_replaced)} secret"
                    f"{'' if len(secrets_replaced) == 1 else 's'} in the supplied config -> "
                    f"{secrets_replaced}"
                )
            if secrets_not_replaced:
                g.logger.debug(
                    f"{lp} there {'is' if len(secrets_not_replaced) == 1 else 'are'} "
                    f"{len(secrets_not_replaced)} secret{'' if len(secrets_not_replaced) == 1 else 's'}"
                    f" configured in the supplied config that {'has' if len(secrets_not_replaced) == 1 else 'have'}"
                    f" no substitution candidate{'' if len(secrets_not_replaced) == 1 else 's'} in the "
                    f"secrets_pool -> {secrets_not_replaced}"
                )
        else:
            g.logger.debug(f"{lp} there are no secrets to grab from, skipping substituting {{[secrets]}}")

        if return_config and isinstance(return_config, str):
            try:
                return_config = literal_eval(return_config)
            except ValueError or TypeError or SyntaxError as e:
                g.logger.error(
                    f"{lp} there is a formatting error in the config file, error converting "
                    f"to a python data structure! Please review your config, remember "
                    f"to always quote the '{{[secrets]}}', '{{{{variables}}}}' and strings that contain "
                    f"special characters '@&^%#$@(*@)(_#&*$%@#%'"
                )
                raise e

        return return_config

    def _parse_conf(
        self,
        config: Optional[dict] = None,
        config_pool: Optional[dict] = None,
        alternate_pool: Optional[dict] = None,
        eval_sections: Optional[set] = None,
    ) -> dict:
        """A wrapper around ``_parse_vars`` that will also literal_eval ``eval_sections`` keys found in the ``config``.



        .. note:: If a substitution variable is replaced by ``alternate_pool`` it will be prepended with '**'.
        :param dict config: (optional) the config to parse, if not supplied, self.config will be used.
        :param dict config_pool: (Optional) the dictionary to use for substitution, if not supplied uses config
        :param dict alternate_pool: (Optional) second dictionary to use for substitution if the first pool does not contain the key
        :param set eval_sections: (Optional) a ``set()`` of sections to evaluate.
        :return: parsed dictionary with {{substitution variables}} replaced with their configured values.
        """
        lp: str = "conf:parse/eval:"
        config = self._parse_vars(config=config, config_pool=config_pool, alternate_pool=alternate_pool)
        if not eval_sections:
            g.logger.debug(f"{lp} 'eval_sections' argument was not supplied, using hardcoded values")
            eval_sections = {
                "pyzm_overrides",
                "platerec_regions",
                "poly_color",
                "hass_people",
                "zmes_keys",
            }

        for e_sec in eval_sections:
            if config.get(e_sec) and isinstance(config[e_sec], str):
                try:
                    config[e_sec] = literal_eval(config[e_sec])
                except ValueError or TypeError or SyntaxError as e:
                    g.logger.error(f"{lp} error converting {e_sec} to a dictionary")
                    raise e

        return config

    def _parse_vars(
        self,
        config: Optional[dict] = None,
        config_pool: Optional[dict] = None,
        alternate_pool: Optional[dict] = None,
    ) -> dict:
        """Method to substitute {{vars}} in the supplied config.

        Search and replace ``config`` for {{substitution variables}}, using supplied ``config_pool`` or ``alternate_pool`` to pull key/value's from.

        .. note:: If a substitution variable is replaced by ``alternate_pool`` it will be prepended with '**' in the logs.
                If ``config`` is not supplied ``self.config`` will be used if it has a value. If ``config_pool`` is not supplied then ``config`` will be ``config_pool``.


        :param config: (Optional) the config to parse, defaults as the config_pool
        :param config_pool: (Optional) the dictionary to use for substitution, if not supplied uses config
        :param dict alternate_pool: (Optional) second dictionary to use for substitution if config_pool does not contain the key
        :return: parsed dictionary with {{substitution variables}} replaced with their configured values.
        """
        # todo: need to test alternate_pool, the config and return_config literal_eval hack
        lp: str = "conf:sub vars:"
        vars_replaced: list = []
        vars_not_replaced: list = []
        config_as_pool: bool = False
        g.logger.debug(f"{lp} starting '{{{{variable}}}}' search and substitution")
        if config is None and self.config:
            config = self.config

        if config_pool:
            g.logger.debug(f"{lp} 'config_pool' argument supplied!")
        else:
            config_as_pool = True
            config_pool = config

        if not config:
            g.logger.warning(
                f"{lp} there is no config supplied to replace {{[substitution variables]}}, skipping "
                f"and returning an empty dictionary"
            )
            return {}
        if alternate_pool is None:
            alternate_pool = {}
        else:
            g.logger.debug(f"{lp} alternate_pool supplied!")

        # For regex to work the dictionary needs to be converted to a string,
        # this is why sub-string replacement works.
        return_config = str(config)
        # Use set to remove duplicates then convert back to a list to allow indexing
        found_sub_vars: list = list(set(compile(r"{{(\w*)}}").findall(return_config)))
        # Make sure base_data_path is processed first or there will be issues, this also allows for base_data_path
        # to be anywhere in the config and still be parsed first for substitutions
        if "base_data_path" in found_sub_vars:
            # .pop() on a list needs the index, hence the .index() call
            t_ = found_sub_vars.pop(found_sub_vars.index("base_data_path"))
            # make it the first var to be parsed and replaced
            found_sub_vars.insert(0, t_)
        eval_needed: bool = False
        for sub_var in found_sub_vars:
            # This is needed if the value is replaced and updated, the config (being used as the key pool)
            # also needs to be updated as config_pool will have all the keys BUT the values will have {{}}, {[]}
            if config_as_pool and eval_needed:
                config = literal_eval(return_config)
                eval_needed = False
            sub_pattern: str = r"(\{{\{{{key}\}}\}})".format(key=sub_var)
            if sub_var in config_pool:
                vars_replaced.append(sub_var)
                if config_as_pool:
                    return_config = compile(sub_pattern).sub(str(config[sub_var]), return_config)
                    eval_needed = True
                else:
                    return_config = compile(sub_pattern).sub(str(config_pool[sub_var]), return_config)
            elif sub_var in alternate_pool:
                # todo: alternate_pool needs testing to figure out if it also needs 'config_as_pool' hack
                # '**' will indicate replaced by alternate_pool
                vars_replaced.append(f"**{sub_var}")
                return_config = compile(sub_pattern).sub(str(alternate_pool[sub_var]), return_config)
            else:
                vars_not_replaced.append(sub_var)

        if vars_replaced:
            g.logger.debug(
                f"{lp} successfully replaced {len(vars_replaced)} sub var"
                f"{'' if len(vars_replaced) == 1 else 's'} -> {vars_replaced}"
            )
        if vars_not_replaced:
            g.logger.debug(
                f"{lp} there {'is' if len(vars_not_replaced) == 1 else 'are'} "
                f"{len(vars_not_replaced)} sub var{'' if len(vars_not_replaced) == 1 else 's'}"
                f" in the supplied config that {'has' if len(vars_not_replaced) == 1 else 'have'} no substitution "
                f"value in the supplied key pools -> {vars_not_replaced}"
            )
        if return_config and isinstance(return_config, str):
            try:
                return_config = literal_eval(return_config)
            except ValueError or TypeError or SyntaxError as e:
                g.logger.error(
                    "something is wrong with the config file formatting, make sure all of your {[secrets]} "
                    "and {{sub vars}} have quotes around them if they are by themselves or in a quoted string if "
                    "it is embedded into it as a sub-string"
                )
                raise e

        return return_config

    def override_monitor(self, mid: int, config: dict) -> dict:
        """Build a config based on the per-monitor section 'monitors'.

        :param dict config: the configuration to use to replace secrets and substitution variables in.
        :param int mid: the monitor ID to build for
        """
        lp: str = f"conf:build mID {mid}:"
        illegal_keys = {
            "base_data_path",
            "mlapi_secret_key",
            "port",
            "processes",
            "db_path",
            "secrets",
            "config",
            "debug",
            "baredebug",
            "version",
            "bareversion",
        }
        if mid not in self.monitors:
            g.logger.warning(
                f"{lp} the requested monitor does not have a per monitor section to build an overridden config"
            )
            return config
        g.logger.debug(f"{lp} attempting to search and replace secrets and sub-vars in per-monitor ({mid}) section")
        # Replace any {{secrets}} or {{vars}} in the 'monitors' section, save it as 'built'
        self.built_monitors[mid] = self._parse_secrets(config=self.monitors[mid], secrets_pool=self.secrets_pool)
        self.built_monitors[mid] = self._parse_vars(config=self.monitors[mid], alternate_pool=config)
        # Convert polygon coords from a string into a proper tuple for shapely.geometry.Polygon to consume
        from pyzm.helpers.pyzm_utils import str2tuple

        # use Polygon to confirm proper coords
        from shapely.geometry import Polygon

        # Check if a key in the per-monitor section is not in the supplied config
        # If it is missing, add it to the supplied config
        # Example: car_min_confidence is not in config, but it is in the per-monitor section.
        overrode: list = []
        new_: list = []
        for overrode_key, overrode_val in self.built_monitors[mid].items():
            if overrode_key in illegal_keys:
                g.logger.warning(
                    f"{lp} can not override '{overrode_key}' in monitor '{mid}' config, "
                    f"this may cause unexpected behavior and is off limits for per monitor overrides"
                )
                continue
            elif overrode_key == "zones":
                g.logger.debug(f"{lp} pre-defined 'zones' found, parsing...")
                zones: dict = overrode_val
                for zone_name, zone_items in zones.items():
                    zone_coords = zone_items.get("coords")
                    zone_pattern = zone_items.get("pattern")
                    zone_contains = zone_items.get("contains")
                    zone_max_size = zone_items.get("max_size")
                    zone_min_conf = zone_items.get("min_conf")
                    if not zone_coords:
                        g.logger.debug(f"{lp} no coords for zone {zone_name}, 'coords' is REQUIRED! skipping...")
                        continue

                    g.logger.debug(f"{lp} polygon specified -> '{zone_name}', validating polygon shape...")
                    try:
                        coords = str2tuple(zone_coords)
                        test = Polygon(coords)
                    except Exception as exc:
                        g.logger.warning(
                            f"{lp} the polygon coordinates supplied from '{overrode_key}' "
                            f"are malformed! -> {overrode_val}, skipping..."
                        )
                        g.logger.debug(f"{lp} EXCEPTION>>> {exc}")
                    else:
                        # Handle append or creating a new entry
                        if mid in self.polygons:
                            g.logger.debug(f"{lp} appending to the existing monitor ID {mid} in polygons!")
                            self.polygons[mid].append(
                                {
                                    "name": zone_name,
                                    "value": coords,
                                    "pattern": zone_pattern,
                                    "contains": zone_contains,
                                    "max_size": zone_max_size,
                                    "min_conf": zone_min_conf,
                                }
                            )
                        else:
                            g.logger.debug(f"{lp} creating new entry in polygons for monitor {mid}")
                            self.polygons[mid] = [
                                {
                                    "name": zone_name,
                                    "value": coords,
                                    "pattern": zone_pattern,
                                    "contains": zone_contains,
                                    "max_size": zone_max_size,
                                    "min_conf": zone_min_conf,
                                }
                            ]
                        g.logger.debug(
                            f"{lp} successfully validated polygon for defined zone '{zone_name}' -> {zone_coords}"
                        )
            elif overrode_key in config and overrode_val is not None:
                overrode.append(overrode_key)

            elif overrode_key not in config and overrode_val is not None:
                # there is not a key to override so a new key will be created
                new_.append(overrode_key)
            config[overrode_key] = overrode_val

        # Add the per-monitor section into the supplied config after substitution
        config["monitors"][mid] = dict(self.built_monitors[mid])
        # Work on the per-monitor config
        config = self._parse_secrets(config=config, secrets_pool=self.secrets_pool)
        # Use the built 'monitors' section as an alternate key pool
        config = self._parse_vars(config=config, alternate_pool=config["monitors"][mid])

        if overrode:
            g.logger.debug(
                f"{lp} {len(overrode)} key{'' if len(overrode) == 1 else 's'} from monitor {mid} per-monitor "
                f"('monitors') section {'was' if len(overrode) == 1 else 'were'} used to override existing keys in "
                f"its own overridden config  -> {overrode}"
            )
        if new_:
            g.logger.debug(
                f"{lp} {len(new_)} key{'' if len(overrode) == 1 else 's'} from monitor {mid} per-monitor ('monitors') "
                f"section {'was' if len(overrode) == 1 else 'were'} added to its own overridden config -> {new_}"
            )
        return config
