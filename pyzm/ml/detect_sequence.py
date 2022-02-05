import re
from ast import literal_eval
from traceback import format_exc
from typing import Optional, List, Dict, AnyStr, Union, Any

import numpy as np
from shapely.geometry import Polygon

import pyzm.ml.alpr as alpr_detect
import pyzm.ml.face as face_detect
from pyzm.helpers.Media import MediaStream
from pyzm.helpers.pyzm_utils import (
    Timer,
    str2bool,
    pkl,
    de_dup,
)
from pyzm.interface import GlobalConfig
from pyzm.ml.aws_rekognition import AwsRekognition
from pyzm.ml.coral_edgetpu import Tpu
from pyzm.ml.yolo import Yolo

lp: str = "detect:"
g: GlobalConfig


def _is_unconverted(expr: str) -> bool:
    """Evaluate a string to see if it is an unconverted secret or substitution variable. If the string begins with
    `{{` or `{[` then it is an unconverted secret or substitution variable.

    >>> _is_unconverted('string')
    """
    expr = expr.strip()
    if expr.startswith("{{"):
        g.logger.debug(f"{expr} seems to be an unconverted substitution variable!")
    elif expr.startswith("{["):
        g.logger.debug(f"{expr} seems to be an unconverted secret!")
    else:
        return True
    return False


class DetectSequence:
    def __init__(self, options: Optional[dict] = None):
        """Initializes ML entry point with various parameters

        Args:
            - options (dict): Variety of ML options. Best described by an example as below

                ::

                    options = {
                        'general': {
                            # sequence of models you want to run for every specified frame
                            'model_sequence': 'object,face,alpr' ,
                            # If 'yes', will not use a BoundedSemaphore lock
                            'disable_locks':'no',

                        },

                        # We now specify all the config parameters per model_sequence entry
                        'object': {
                            'general':{
                                # 'first' - When detecting objects, if there are multiple fallbacks, break out
                                # the moment we get a match using any object detection library.
                                # 'most' - run through all libraries, select one that has most object matches
                                # 'most_unique' - run all models, select one that has most unique object matches
                                # 'union' - sum all the detected objects together between models

                                'same_model_sequence_strategy': 'first' # 'first' 'most', 'most_unique', 'union'
                                'pattern': '.*' # any pattern
                            },

                            # within object, this is a sequence of object detection libraries. In this case,
                            # I want to first try on my TPU and if it fails, try GPU
                            'sequence': [{
                                #First run on TPU
                                'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/
                                ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
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
                                'object_processor': 'gpu',
                                # OPTIONAL: Default is 416. Change if your model is trained for larger sizes
                                'model_width': 416,
                                'model_height': 416
                            }]
                        },

                        # We repeat the same exercise with 'face' as it is next in model_sequence
                        'face': {
                            'general':{
                                'same_model_sequence_strategy': 'first'
                            },
                            'sequence': [{
                                # if max_size is specified, not matter what
                                # the resize value in stream_options, it will be rescaled down to this
                                # value if needed
                                'max_size':800,
                                'face_detection_framework': 'dlib',
                                'known_images_path': '/var/lib/zmeventnotification/known_faces',
                                'face_model': 'cnn',
                                'face_train_model': 'cnn',
                                'face_recog_dist_threshold': 0.6,
                                'face_num_jitters': 1,
                                'face_upsample_times':1
                            }]
                        },

                        # We repeat the same exercise with 'alpr' as it is next in model_sequence
                        'alpr': {
                            'general':{
                                'same_model_sequence_strategy': 'first',

                                # This can be applied to any model. This means, don't run this model
                                # unless a previous model detected one of these labels.
                                # In this case, I'm not calling ALPR unless we've detected a vehile
                                # bacause platerec has an API threshold

                                'pre_existing_labels':['car', 'motorbike', 'bus', 'truck', 'boat'],

                            },
                            'sequence': [{
                                'alpr_api_type': 'cloud',
                                'alpr_service': 'plate_recognizer',
                                'alpr_key': g.config['alpr_key'],
                                'platrec_stats': 'no',
                                'platerec_min_dscore': 0.1,
                                'platerec_min_score': 0.2,
                            }]
                        }
                    } # ml_options



        """
        global g
        g = GlobalConfig()
        if options is None:
            options = {}
        self.has_rescaled: bool = False  # only once in init
        self.model_sequence: Optional[list[str]] = None
        self.ml_options: dict = {}
        self.stream_options: Optional[dict] = None
        self.media: Optional[MediaStream] = None
        self.ml_overrides: dict = {}
        self.models: Dict[AnyStr, List[Union[Tpu, AwsRekognition, Yolo, face_detect, alpr_detect]]] = {}
        self.model_name: str = ""
        self.model_valid: bool = True
        self.set_ml_options(options, force_reload=True)

    def get_ml_options(self) -> dict:
        """Return the ml_options (ml_sequence) dict"""
        return self.ml_options

    def set_ml_options(self, options: dict, force_reload: bool = False):
        """Use this to change ml options later. Note that models will not be reloaded
        unless you add force_reload=True
        """
        lp: str = "detect:set ml opts:"
        if force_reload:
            if self.models:
                g.logger.debug(f"{lp} resetting the loaded models!")
            self.models = {}
        if not options:
            return
        model_sequence: Optional[str] = options.get("general", {}).get("model_sequence")
        # convert model_sequence to a list if it is a string
        if model_sequence and isinstance(model_sequence, str):
            self.model_sequence = model_sequence.split(",")
            self.model_sequence = [x.strip() for x in self.model_sequence]
        else:
            raise ValueError(
                f"model_sequence ({model_sequence}) must be a comma seperated string, current type is "
                f"{type(model_sequence)}"
            )
        self.ml_options: dict = options
        self.stream_options: Optional[dict] = None
        self.media: Optional[MediaStream] = None
        self.ml_overrides: dict = {}

    def _load_models(self, models=None):
        def get_correct_object_framework(frame_work):
            if frame_work == "opencv":
                return Yolo(options=model_sequence)
            elif frame_work == "coral_edgetpu":
                return Tpu(options=model_sequence)
                # AWS Rekognition
            elif frame_work == "aws_rekognition":
                return AwsRekognition(options=model_sequence)
            else:
                raise ValueError(
                    f"Invalid object_framework: {frame_work}. Only opencv, coral_edgetpu "
                    f"(pycoral) and aws_rekognition supported as of now"
                )

        lp: str = "detect:load models:"
        if not isinstance(models, list):
            g.logger.error(f"{lp} models supplied must be a list of models (object, face, alpr)")
        if not models:
            # convert model_sequence to a list if it is a string
            if self.model_sequence and isinstance(self.model_sequence, str):
                models = self.model_sequence.split(",")
            else:
                models = self.model_sequence

        accepted_models: tuple = ("object", "face", "alpr")
        for model in models:
            if model in accepted_models:
                # create an empty list in the model dict for current valid model
                self.models[model] = []
                sequences = self.ml_options.get(model, {}).get("sequence")
                if sequences:
                    ndx = 0
                    for model_sequence in sequences:
                        ndx += 1
                        obj = None
                        seq_name = model_sequence.get("name", f"sequence:{ndx}")
                        if not str2bool(model_sequence.get("enabled", "yes")):
                            g.logger.debug(
                                2,
                                f"{lp} skipping sequence '{seq_name}' ({ndx} of "
                                f"{len(sequences)}) as it is disabled",
                            )
                            continue
                        model_sequence["disable_locks"] = self.ml_options.get("general", {}).get("disable_locks", "no")
                        g.logger.debug(
                            2,
                            f"{lp} loading '{model}' sequence '{seq_name}' ({ndx} of {len(sequences)})",
                        )
                        try:
                            if model == "object":
                                framework = model_sequence.get("object_framework")
                                obj = get_correct_object_framework(framework)
                            elif model == "alpr":
                                obj = alpr_detect.Alpr(options=model_sequence, globs=g)
                            elif model == "face":
                                obj = face_detect.Face(options=model_sequence, globs=g)
                            self.models[model].append(obj)
                        except Exception as exc:
                            g.logger.error(
                                f"{lp} error while trying to construct '{model}' sequence "
                                f"pipeline for '{seq_name}', skipping..."
                            )
                            g.logger.error(f"{lp} EXCEPTION>>>> {exc}")
                            g.logger.debug(format_exc())
                else:
                    g.logger.warning(
                        f"{lp} '{model}' configured, but there are NO 'sequence' sections defined! skipping..."
                    )
            else:
                g.logger.error(f"{lp} the configured model '{model}' is not a valid type! skipping...")

    @staticmethod
    def rescale_polygons(polygons, x_factor, y_factor, method="down"):
        new_ps = []
        for p in polygons:
            # from pyzm.helpers.pyzm_utils import str2tuple
            # coords = str2tuple(p["value"])
            # p['value'] = coords
            newp = []
            for x, y in p["value"]:
                if method == "down":
                    new_x = int(x * x_factor)
                    new_y = int(y * y_factor)
                elif method == "up":
                    new_x = int(x / x_factor)
                    new_y = int(y / y_factor)
                else:
                    g.logger.error(f"{lp} rescale polygons -> can only use 'up' or 'down' for rescaling")
                    return
                newp.append((new_x, new_y))
            new_ps.append({"name": p["name"], "value": newp, "pattern": p["pattern"]})
        g.logger.debug(
            2,
            f"{lp} rescaled polygons ('{method}') using factors: x={x_factor:.4f}  y={y_factor:.4f}\n*** OLD:"
            f" [{polygons}] \n****** NEW:[{new_ps}]",
        )
        return new_ps

    @staticmethod
    def _bbox2poly(bbox):
        """Convert bounding box coords to a Polygon acceptable input for shapely."""
        it = iter(bbox)
        bbox = list(zip(it, it))
        bbox.insert(1, (bbox[1][0], bbox[0][1]))
        bbox.insert(3, (bbox[0][0], bbox[2][1]))
        return bbox
        # return Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])

    # todo: clean up (kw)args
    def _filter_detections(
        self,
        model_name,
        box,
        label,
        conf,
        polygons,
        h,
        w,
        model_names,
        seq_opt=None,
        model_type=None,
        pkl_data=None,
    ):
        """INTERNAL METHOD. Filter out detected objects based upon configured options."""

        # Try to validate options
        if not seq_opt:
            raise ValueError(f"_filter_detections -> seq_opt (sequence options) is Empty or None!")
        elif not self.ml_options:
            raise ValueError(f"_filter_detections -> ml_options (machine learning options) is Empty or None!")
        elif "general" not in self.ml_options:
            raise ValueError(f"_filter_detections -> ml_options:general is not configured!")

        saved_ls: Optional[List[str]] = None
        saved_bs: Optional[List[str]] = None
        saved_cs: Optional[List[str]] = None
        saved_event: Optional[str] = None
        mpd: Optional[Union[str, bool]] = None
        mpd_b: list = []
        mpd_l: list = []
        mpd_c: list = []
        if pkl_data:
            saved_bs = pkl_data["saved_bs"]
            saved_ls = pkl_data["saved_ls"]
            saved_cs = pkl_data["saved_cs"]
            saved_event = pkl_data["saved_event"]
            mpd = pkl_data["mpd"]
            mpd_b = pkl_data["mpd_b"]
            mpd_l = pkl_data["mpd_l"]
            mpd_c = pkl_data["mpd_c"]
        tot_labels: int = len(label) or 0
        max_object_area: Optional[Union[str, float]] = None
        contained_area: float = 10.0
        failed: bool = False
        min_conf: Optional[Union[str, float]] = None
        lp: str = "detect:filtering:"
        # g.logger.debug(4, f"|--- Filtering {tot_labels} {'detections' if tot_labels > 1 else 'detection'} ---|")
        new_label, new_bbox, new_conf, new_err, new_model_names = [], [], [], [], []
        new_bbox_to_poly, error_bbox_to_poly = [], []
        moa: Optional[str] = None
        ioa: Optional[str] = None
        min_conf_found: str = ""
        ioa_found: str = ""
        moa_found: str = ""
        per_keys_: tuple = (
            "object_min_confidence",
            "max_detection_size",
            "contained_area",
            "confidence_upper",
        )
        conf_upper: Optional[Union[str, int, float]] = None
        conf_upper_found: str = ""
        conf_upper_break: bool = False
        for key_ in per_keys_:
            if key_ in seq_opt and seq_opt[key_] is not None and not seq_opt[key_].startswith("{{"):
                if key_ == "object_min_confidence":
                    min_conf = seq_opt[key_]
                    min_conf_found = f"Sequence {seq_opt.get('name')} -> {key_}"
                elif key_ == "confidence_upper":
                    conf_upper = seq_opt[key_]
                    conf_upper_found = f"Sequence {seq_opt.get('name')} -> {key_}"
                elif key_ == "max_detection_size":
                    moa = seq_opt[key_]
                    moa_found = f"Sequence {seq_opt.get('name')} -> {key_}"
                elif key_ == "contained_area":
                    ioa = seq_opt[key_]
                    ioa_found = f"Sequence {seq_opt.get('name')} -> {key_}"
            elif (
                key_ in self.ml_options.get("general")
                and self.ml_options.get("general")[key_] is not None
                and not self.ml_options.get("general")[key_].startswith("{{")
            ):
                if key_ == "object_min_confidence":
                    min_conf = self.ml_options.get("general").get(key_)
                    min_conf_found = f"ml_sequence:general -> {key_}"
                elif key_ == "confidence_upper":
                    conf_upper = self.ml_options.get("general").get(key_)
                    conf_upper_found = f"ml_sequence:general -> {key_}"
                elif key_ == "max_detection_size":
                    moa = self.ml_options.get("general").get(key_)
                    moa_found = f"ml_sequence:general -> {key_}"
                elif key_ == "contained_area":
                    ioa = self.ml_options.get("general").get(key_)
                    ioa_found = f"ml_sequence:general -> {key_}"
            elif (
                key_ in self.ml_options.get(model_name, {}).get("general")
                and self.ml_options.get(model_name, {}).get("general")[key_] is not None
                and not self.ml_options.get(model_name, {}).get("general")[key_].startswith("{{")
            ):
                if key_ == "object_min_confidence":
                    min_conf = self.ml_options.get(model_name, {}).get("general").get(key_)
                    min_conf_found = f"Model {model_name}:general -> {key_}"
                elif key_ == "confidence_upper":
                    conf_upper = self.ml_options.get(model_name, {}).get("general").get(key_)
                    conf_upper_found = f"Model {model_name}:general -> {key_}"
                elif key_ == "max_detection_size":
                    moa = self.ml_options.get(model_name, {}).get("general").get(key_)
                    moa_found = f"Model {model_name}:general -> {key_}"
                elif key_ == "contained_area":
                    ioa = self.ml_options.get(model_name, {}).get("general").get(key_)
                    ioa_found = f"Model {model_name}:general -> {key_}"
        appended: bool = False

        for idx, b in enumerate(box):
            lp = "detect:filter:"
            if failed:
                # for each object that failed before loop end
                g.logger.debug(
                    2,
                    f"--- '{label[idx - 1]} ({idx}/{tot_labels})' has FAILED filtering",
                )
                failed = False
            elif conf_upper_break and appended:
                g.logger.debug(
                    f"confidence_upper: the configured limit has been hit and a filtered "
                    f"match is present, short-circuiting!"
                )
                break
            elif conf_upper_break and not appended:
                conf_upper_break = False
                g.logger.debug(f"confidence_upper: the configured limit was hit BUT the object was filtered out")
            appended = False
            show_label = f"{label[idx]} ({idx + 1}/{tot_labels})"
            g.logger.debug(f">>> detected '{show_label}' confidence: {conf[idx]:.2f}")
            poly_b = self._bbox2poly(b)
            # save b as the objects polygon
            obj = Polygon(poly_b)

            # Per label overrides
            per_keys_ = (
                "_object_min_confidence",
                "_max_detection_size",
                "_contained_area",
                "_confidence_upper",
            )
            for key_ in per_keys_:
                if (
                    seq_opt.get(f"{label[idx]}{key_}")
                    and seq_opt[f"{label[idx]}{key_}"] is not None
                    and not seq_opt[f"{label[idx]}{key_}"].startswith("{{")
                ):
                    if key_ == "_object_min_confidence":
                        min_conf = seq_opt.get(f"{label[idx]}{key_}")
                        min_conf_found = f"Sequence {seq_opt.get('name')} -> {label[idx]}{key_}"
                    elif key_ == "_confidence_upper":
                        conf_upper = seq_opt.get(f"{label[idx]}{key_}")
                        conf_upper_found = f"Sequence {seq_opt.get('name')} -> {label[idx]}{key_}"
                    elif key_ == "_max_detection_size":
                        moa = seq_opt.get(f"{label[idx]}{key_}")
                        moa_found = f"Sequence {seq_opt.get('name')} -> {label[idx]}{key_}"
                    elif key_ == "_contained_area":
                        ioa = seq_opt.get(f"{label[idx]}{key_}")
                        ioa_found = f"Sequence {seq_opt.get('name')} -> {label[idx]}{key_}"
                elif (
                    self.ml_options.get("general").get(f"{label[idx]}{key_}")
                    and self.ml_options.get("general").get(f"{label[idx]}{key_}") is not None
                    and not self.ml_options.get("general").get(f"{label[idx]}{key_}").startswith("{{")
                ):
                    if key_ == "_min_confidence":
                        min_conf = self.ml_options.get("general").get(key_)
                        min_conf_found = f"ml_sequence:general -> {label[idx]}{key_}"
                    elif key_ == "_confidence_upper":
                        conf_upper = self.ml_options.get("general").get(key_)
                        conf_upper_found = f"ml_sequence:general -> {label[idx]}{key_}"
                    elif key_ == "_max_detection_size":
                        moa = self.ml_options.get("general").get(key_)
                        moa_found = f"ml_sequence:general -> {label[idx]}{key_}"
                    elif key_ == "_contained_area":
                        ioa = self.ml_options.get("general").get(key_)
                        ioa_found = f"ml_sequence:general -> {label[idx]}{key_}"
                elif (
                    self.ml_options.get(model_name, {}).get("general").get(f"{label[idx]}{key_}")
                    and self.ml_options.get(model_name, {}).get("general").get(f"{label[idx]}{key_}") is not None
                    and not self.ml_options.get(model_name, {})
                    .get("general")
                    .get(f"{label[idx]}{key_}")
                    .startswith("{{")
                ):
                    if key_ == "_object_min_confidence":
                        min_conf = self.ml_options.get(model_name, {}).get("general").get(f"{label[idx]}{key_}")
                        min_conf_found = f"Model {model_name}:general -> {label[idx]}{key_}"
                    elif key_ == "_confidence_upper":
                        conf_upper = self.ml_options.get(model_name, {}).get("general").get(f"{label[idx]}{key_}")
                        conf_upper_found = f"Model {model_name}:general -> {label[idx]}{key_}"
                    elif key_ == "_max_detection_size":
                        moa = self.ml_options.get(model_name, {}).get("general").get(f"{label[idx]}{key_}")
                        moa_found = f"Model {model_name}:general -> {label[idx]}{key_}"
                    elif key_ == "_contained_area":
                        ioa = self.ml_options.get(model_name, {}).get("general").get(f"{label[idx]}{key_}")
                        ioa_found = f"Model {model_name}:general -> {label[idx]}{key_}"

            pattern_match = None
            p_mpd: Optional[dict] = None
            mda: Optional[str] = None
            mda_found: str = ""
            for p in polygons:
                # defined 'zones'
                p_ioa: Optional[dict] = p.get("contains", {})
                p_moa: Optional[dict] = p.get("max_size", {})
                p_min_conf: Optional[dict] = p.get("min_conf", {})
                p_mpd: Optional[dict] = p.get("past_area_diff", {})
                p_conf_upper: Optional[dict] = p.get("conf_upper", {})

                p_ioa = {} if p_ioa is None else p_ioa
                p_moa = {} if p_moa is None else p_moa
                p_min_conf = {} if p_min_conf is None else p_min_conf
                p_mpd = {} if p_mpd is None else p_mpd
                p_conf_upper = {} if p_min_conf is None else p_conf_upper
                if "all" in p_conf_upper:
                    conf_upper = p["conf_upper"]["all"]
                    conf_upper_found = f"Defined Zone:{p.get('name')} -> ALL"
                elif label[idx] in p_conf_upper:
                    conf_upper = p["conf_upper"][label[idx]]
                    conf_upper_found = f"Defined Zone:{p.get('name')} -> {label[idx]}"
                if "all" in p_mpd:
                    mda = p["past_area_diff"]["all"]
                    mda_found = f"Defined Zone:{p.get('name')} -> ALL"
                elif label[idx] in p_mpd:
                    mda = p["past_area_diff"][label[idx]]
                    mda_found = f"Defined Zone:{p.get('name')} -> {label[idx]}"
                if "all" in p_ioa:
                    ioa = p.get("contains").get("all")
                    ioa_found = f"Defined Zone:{p.get('name')} -> ALL"
                elif label[idx] in p_ioa:
                    ioa = p.get("contains").get(label[idx])
                    ioa_found = f"Defined Zone:{p.get('name')} -> {label[idx]}"
                if "all" in p_moa:
                    moa = p.get("max_size").get("all")
                    moa_found = f"Defined Zone:{p.get('name')} -> ALL"
                elif label[idx] in p_moa:
                    moa = p.get("max_size").get(label[idx])
                    moa_found = f"Defined Zone:{p.get('name')}:{label[idx]}:"
                if "all" in p_min_conf:
                    min_conf = p["min_conf"]["all"]
                    min_conf_found = f"Defined Zone:{p.get('name')} -> ALL"
                elif label[idx] in p_min_conf:
                    min_conf = p.get("min_conf").get(label[idx])
                    min_conf_found = f"Defined Zone:{p.get('name')}:{label[idx]}:"

                if not min_conf:
                    # min_conf IS REQUIRED
                    g.logger.warning(f"{lp} {show_label} min_conf not found! Using 50%")
                    min_conf = 0.5
                    min_conf_found = "NOT FOUND - DEFAULT->50%"
                if min_conf:
                    # Allow for [0*]0.XX , 34 or 34% input for confidence - 34 would be evaluated as 34%
                    _m = re.match(r"(0*?\.?\d*)(%)?$", str(min_conf), re.IGNORECASE)
                    if _m:
                        try:
                            starts_with: Optional[re.Match] = None
                            if _m.group(1):
                                starts_with = re.search(r"(0*\.?)(\d*)(%)?$", _m.group(1), re.IGNORECASE)
                            if _m.group(2) == "%":
                                # Explicit %
                                min_conf = float(_m.group(1)) / 100.0
                            elif starts_with and not starts_with.group(1):
                                # there is no % at end and the string does not start with 0*. or .
                                # consider it a percentile input
                                min_conf = float(_m.group(1)) / 100.0
                            else:
                                min_conf = float(_m.group(1))
                        except TypeError or ValueError:
                            g.logger.warning(f"{lp} {show_label} min_conf could not be converted to a FLOAT! Using 50%")
                            min_conf = 0.5
                            min_conf_found = "FLOAT ERROR - DEFAULT->50%"
                    else:
                        g.logger.warning(f"{lp} {show_label} minimum confidence malformed! ({min_conf}) Using 50%")
                        min_conf = 0.5
                        min_conf_found = "MALFORMED - DEFAULT->50%"
                    g.logger.debug(f"'{show_label}' minimum confidence found: ({min_conf_found}) -> '{min_conf}'")
                    if conf[idx] < min_conf:  # confidence filter
                        g.logger.debug(
                            2,
                            f"confidence: {conf[idx] * 100:.2f} is lower than minimum of {min_conf * 100:.2f}, "
                            f"removing...",
                        )
                        # don't draw red error boxes around filtered out objects by conf if not specified in config
                        if str2bool(g.config.get("show_conf_filtered")):
                            error_bbox_to_poly.append(b)
                            new_err.append(b)
                        failed = True
                        continue
                if conf_upper:
                    # Allow for 0.XX , 34 or 34% input for confidence - 34 would be evaluated as 34%
                    _m = re.match(r"(\d*\.?\d*)(%)?$", str(conf_upper), re.IGNORECASE)
                    if _m:
                        try:
                            starts_with: Optional[re.Match] = None
                            if _m.group(1):
                                starts_with = re.search(r"(0*\.?)(\d*)(%)?$", _m.group(1), re.IGNORECASE)
                            if _m.group(2) == "%":
                                # Explicit %
                                conf_upper = float(_m.group(1)) / 100.0
                            elif starts_with and not starts_with.group(1):
                                # there is no % at end and the string does not start with 0*. or .
                                # consider it a percentile input
                                conf_upper = float(_m.group(1)) / 100.0
                            else:
                                conf_upper = float(_m.group(1))
                        except TypeError or ValueError:
                            g.logger.warning(
                                f"{lp} {show_label} confidence_upper could not be converted to a FLOAT! "
                                f"skipping this filter"
                            )
                        else:
                            g.logger.debug(
                                f"'{show_label}' UPPER confidence found: ({conf_upper_found}) -> '{conf_upper}'"
                            )
                            if conf[idx] >= conf_upper:  # upper confidence filter (satisfactory/good enough logic)
                                g.logger.debug(
                                    2,
                                    f"confidence: {conf[idx] * 100:.2f} is equal to or higher than "
                                    f"{conf_upper * 100:.2f}, accepting as a match..",
                                )
                                conf_upper_break = True

                if moa and moa != "100%":
                    g.logger.debug(
                        4,
                        f"'{show_label}' max area of detected object found ({moa_found}) -> '{moa}'",
                    )
                    # Let's make sure it's the right size
                    _m = re.match(r"(\d*\.?\d*)(px|%)?$", moa, re.IGNORECASE)
                    if _m:
                        max_object_area = float(_m.group(1))
                        if _m.group(2) == "%":
                            max_object_area = float(_m.group(1)) / 100.0 * (h * w)

                            g.logger.debug(
                                2,
                                f"max size: converted {_m.group(1)}% of {w}*{h}->{round(w * h)} to "
                                f"{max_object_area} pixels",
                            )
                            if max_object_area and max_object_area > (h * w):
                                max_object_area = h * w
                    if max_object_area and obj.area > max_object_area:
                        g.logger.debug(
                            f"max size: {obj.area:.2f} is larger then the max allowed: {max_object_area:.2f},"
                            f"removing..."
                        )
                        failed = True
                        error_bbox_to_poly.append(b)
                        new_err.append(b)
                        continue

                if p["name"] != "full_image":
                    polygon_zone = Polygon(p["value"])
                    g.logger.debug(
                        2,
                        f"checking if '{show_label}' @ {b} is inside polygon/zone '{p['name']}' located at {p['value']}",
                    )
                    if obj.intersects(polygon_zone):
                        g.logger.debug(f"'{show_label}' INTERSECTS polygon/zone '{p['name']}'")
                        pixels_inside = obj.intersection(polygon_zone).area
                        percent_inside = (pixels_inside / obj.area) * 100
                        g.logger.debug(
                            2,
                            f"'{show_label}' has {pixels_inside:.2f} pixels ({percent_inside:.2f}%) inside"
                            f" '{p['name']}'",
                        )
                        if ioa:
                            g.logger.debug(
                                4,
                                f"minimum area of detected object inside of polygon/zone: found ({ioa_found})"
                                f" -> '{ioa}'",
                            )
                            # Let's make sure it's the right size
                            _m = re.match(r"(\d*\.?\d*)(px|%)?$", ioa, re.IGNORECASE)
                            if _m:
                                contained_area = float(_m.group(1))
                                if _m.group(2) == "%":
                                    contained_area = float(_m.group(1)) / 100.0 * obj.area
                                    g.logger.debug(
                                        2,
                                        f"contained within zone: converted {_m.group(1)}% of {show_label}'s area "
                                        f"({obj.area}) to {contained_area} pixels",
                                    )
                        if pixels_inside < contained_area:
                            g.logger.debug(
                                f"'{show_label}' only has {pixels_inside:.2f} pixels ({percent_inside:.2f}%) inside "
                                f"of {p.get('name')} minimum is -> {contained_area} ("
                                f"{round((contained_area / obj.area) * 100, 2)}%)"
                            )
                            error_bbox_to_poly.append(b)
                            new_err.append(b)
                            failed = True
                            continue
                    else:
                        error_bbox_to_poly.append(b)
                        new_err.append(b)
                        g.logger.debug(
                            2,
                            f"intersection: '{show_label}' does not intersect zone: {p['name']}, removing...",
                        )
                        failed = True
                        continue

                # pattern matching is here because polygon/zone might have its own match pattern
                if str2bool(self.ml_overrides.get("enable")) and self.ml_overrides.get(model_name, {}).get(
                    f"{model_type}_detection_pattern"
                ):
                    match_pattern = self.ml_overrides.get(model_name, {}).get(f"{model_type}_detection_pattern")
                    g.logger.debug(
                        2,
                        "match pattern: overridden by ml_overrides from '{}' to '{}'".format(
                            self.ml_options.get(model_name, {})
                            .get("general", {})
                            .get("{}_detection_pattern".format(model_type), ".*"),
                            match_pattern,
                        ),
                    )
                elif p["pattern"]:
                    g.logger.debug(
                        3,
                        "match pattern: '{}' "
                        "zone '{}' has overrides->'{}'".format(show_label, p["name"], p["pattern"]),
                    )
                    match_pattern = p["pattern"]
                else:
                    match_pattern = (
                        self.ml_options.get(model_name, {})
                        .get("general", {})
                        .get("{}_detection_pattern".format(model_type), ".*")
                    )
                    g.logger.debug(2, f"match pattern: {match_pattern}")

                r = re.compile(match_pattern)
                match = list(filter(r.match, label))

                if label[idx] not in match:
                    error_bbox_to_poly.append(b)
                    new_err.append(b)
                    g.logger.debug(
                        3,
                        f"match pattern: '{show_label}' does not match pattern filter, removing...",
                    )
                    continue
                elif label[idx] in match:
                    pattern_match = True
            # out of polygon/zone loop
            if not pattern_match:
                failed = True
                continue

            # MATCH PAST DETECTIONS
            # Allows match_past_detections to be enabled using sequence options - WIP
            seq_mpd = seq_opt.get("match_past_detections")
            if (
                (str2bool(mpd) or str2bool(seq_mpd))
                and (
                    not g.config.get("PAST_EVENT")
                    or (g.config.get("PAST_EVENT") and str2bool(g.config.get("force_mpd")))
                )
                and ((not g.eid == saved_event) or (g.eid == saved_event and str2bool(g.config.get("force_mpd"))))
            ):
                lp = "mpd:"
                if not saved_bs:
                    g.logger.debug(
                        f"{lp} there are no saved detections to evaluate, skipping match past detection filter"
                    )
                else:

                    max_diff_area: Optional[Union[str, float]]
                    use_percent: bool = False
                    ignore_mpd: bool = False
                    # Check if the pre-defined zone filter was already set as it takes precedence
                    if not p_mpd and not mda:
                        # Start in the general section of ml_sequence
                        if self.ml_options.get("general", {}).get("past_det_max_diff_area"):
                            if _is_unconverted(self.ml_options["general"]["past_det_max_diff_area"]):
                                mda = self.ml_options["general"]["past_det_max_diff_area"]
                                mda_found = f"ml_sequence:general -> past_det_max_diff_area"
                        if self.ml_options.get("general", {}).get(f"{label[idx]}_past_det_max_diff_area"):
                            if _is_unconverted(self.ml_options["general"][f"{label[idx]}_past_det_max_diff_area"]):
                                mda = self.ml_options["general"][f"{label[idx]}_past_det_max_diff_area"]
                                mda_found = f"ml_sequence:general -> {label[idx]}_past_det_max_diff_area"
                        # MODEL_NAME>general
                        if self.ml_options.get(model_name, {}).get("general", {}).get("past_det_max_diff_area"):
                            if _is_unconverted(self.ml_options[model_name]["general"]["past_det_max_diff_area"]):
                                mda = self.ml_options[model_name]["general"]["past_det_max_diff_area"]
                                mda_found = f"Model {model_name}:general -> past_det_max_diff_area"
                        if (
                            self.ml_options.get(model_name, {})
                            .get("general", {})
                            .get(f"{label[idx]}_past_det_max_diff_area")
                        ):
                            if _is_unconverted(
                                self.ml_options[model_name]["general"][f"{label[idx]}_past_det_max_diff_area"]
                            ):
                                mda = self.ml_options[model_name]["general"][f"{label[idx]}_past_det_max_diff_area"]
                                mda_found = f"Model {model_name}:general -> {label[idx]}_past_det_max_diff_area"
                        # Sequence options
                        if seq_opt.get("past_det_max_diff_area"):
                            mda = seq_opt.get("past_det_max_diff_area")
                            mda_found = f"Sequence {seq_opt.get('name')} -> past_det_max_diff_area"
                        if seq_opt.get(f"{label[idx]}_past_det_max_diff_area"):
                            mda = seq_opt.get(f"{label[idx]}_past_det_max_diff_area")
                            mda_found = f"Sequence {seq_opt.get('name')} -> {label[idx]}_past_det_max_diff_area"
                    # Check if the mpd_ignore option is configured in general or in the per sequence
                    mpd_ig = self.ml_options.get("general", {}).get("ignore_past_detection_labels")
                    # If it is still a string it needs to be evaluated into a list
                    if isinstance(mpd_ig, str) and len(mpd_ig):
                        try:
                            mpd_ig = literal_eval(mpd_ig)
                            if not isinstance(mpd_ig, list):
                                g.logger.warning(
                                    f"{lp} ignore_past_detection_labels is not a list after evaluation! "
                                    f"raising ValueError"
                                )
                                raise ValueError("ignore_past_detection_labels is malformed!")
                        except ValueError or SyntaxError as e:
                            g.logger.warning(
                                f"{lp} ignore_past_detection_labels ("
                                f"{self.ml_options['general']['ignore_past_detection_labels']}) "
                                f"is malformed, ignoring..."
                            )
                            g.logger.debug(f"{lp} ignore_past_detection_labels EXCEPTION MESSAGE: {e}")
                            mpd_ig = []
                    if mpd_ig and label[idx] in mpd_ig:
                        g.logger.debug(
                            4,
                            f"{lp} {label[idx]} is in ignore list: {mpd_ig}, skipping",
                        )
                        ignore_mpd = True

                    if not ignore_mpd:
                        g.logger.debug(f"{lp} max detection area found! ({mda_found}) -> {mda}")
                        # Format max difference area
                        if mda:
                            # _m = re.match(r"(\d+)(px|%)?$", mda, re.IGNORECASE)
                            _m = re.match(r"(\d*\.?\d*)(px|%)?$", mda, re.IGNORECASE)

                            if _m:
                                max_diff_area = float(_m.group(1))
                                use_percent = True if _m.group(2) is None or _m.group(2) == "%" else False
                            else:
                                g.logger.error(f"{lp}  malformed -> {mda}, setting to 5%...")
                                use_percent = True
                                max_diff_area = 5.0

                            # it's very easy to forget to add 'px' when using pixels
                            if use_percent and (max_diff_area < 0 or max_diff_area > 100):
                                g.logger.error(
                                    f"{lp} {max_diff_area} must be in the range 0-100 when "
                                    f"using percentages: {mda}, setting to 5%..."
                                )
                                max_diff_area = 5.0
                        else:
                            g.logger.debug(
                                f"{lp} no past_det_max_diff_area or per label overrides configured while "
                                f"match_past_detections=yes, setting to 5% as default"
                            )
                            max_diff_area = 5.0
                            use_percent = True

                        g.logger.debug(
                            4,
                            f"{lp} max difference in area configured ({mda_found}) -> '{mda}', comparing past "
                            f"detections to current",
                        )

                        # Compare current detection to past detections AREA
                        for saved_idx, saved_b in enumerate(saved_bs):
                            # compare current detection element with saved list from file
                            found_past_match = False
                            aliases: Union[str, dict] = self.ml_options.get("general", {}).get("aliases", {})
                            # parse from a str into a dict
                            if isinstance(aliases, str) and len(aliases):
                                aliases = literal_eval(aliases)
                            if saved_ls[saved_idx] != label[idx]:
                                if aliases and isinstance(aliases, dict):
                                    g.logger.debug(f"{lp} checking aliases")
                                    for item, value in aliases.items():
                                        if found_past_match:
                                            break
                                        elif saved_ls[saved_idx] in value and label[idx] in value:
                                            found_past_match = True
                                            g.logger.debug(
                                                2,
                                                f"{lp} aliases: found current label -> '{label[idx]}' and past label "
                                                f"-> '{saved_ls[saved_idx]}' are in an alias group named -> '{item}'",
                                            )
                                elif aliases and not isinstance(aliases, dict):
                                    g.logger.debug(
                                        f"{lp} aliases are configured but the format is incorrect, check the example "
                                        f"config for formatting and reformat aliases to a dictionary type setup"
                                    )

                            elif saved_ls[saved_idx] == label[idx]:
                                found_past_match = True
                            if not found_past_match:
                                continue
                            # Found a match by label, now compare the area using Polygon
                            saved_poly = self._bbox2poly(saved_b)
                            saved_obj = Polygon(saved_poly)
                            max_diff_pixels = None
                            g.logger.debug(
                                4,
                                f"{lp} comparing '{label[idx]}' PAST->{saved_b} to CURR->{b}",
                            )
                            if saved_obj.intersects(obj) or obj.intersects(saved_obj):
                                if saved_obj.intersects(obj):
                                    g.logger.debug(
                                        4,
                                        f"{lp} the PAST object INTERSECTS the new object",
                                    )
                                else:
                                    g.logger.debug(
                                        4,
                                        f"{lp} the current object INTERSECTS the PAST object",
                                    )
                                diff_area = None
                                if obj.contains(saved_obj):
                                    diff_area = obj.difference(saved_obj).area
                                    if use_percent:
                                        max_diff_pixels = obj.area * max_diff_area / 100
                                else:
                                    diff_area = saved_obj.difference(obj).area
                                    if use_percent:
                                        max_diff_pixels = saved_obj.area * max_diff_area / 100
                                if diff_area is not None and diff_area <= max_diff_pixels:
                                    g.logger.debug(
                                        f"{lp} removing '{show_label}' as it seems to be in the same spot as it was "
                                        f"detected last time based on '{mda}' -> Difference in pixels: {diff_area} "
                                        f"- Configured maximum difference in pixels: {max_diff_pixels}"
                                    )
                                    if saved_b not in mpd_b:
                                        g.logger.debug(
                                            f"{lp} appending this saved object to the mpd "
                                            f"buffer as it has removed a detection and should be propagated "
                                            f"to the next event"
                                        )
                                        mpd_b.append(saved_bs[saved_idx])
                                        mpd_l.append(saved_ls[saved_idx])
                                        mpd_c.append(saved_cs[saved_idx])
                                    new_err.append(b)
                                    failed = True
                                    break
                                elif diff_area is not None and diff_area > max_diff_pixels:
                                    g.logger.debug(
                                        4,
                                        f"{lp} allowing '{show_label}' -> the difference in the area of last detection "
                                        f"to this detection is '{diff_area:.2f}', a minimum of {max_diff_pixels:.2f} "
                                        f"is needed to not be considered 'in the same spot'",
                                    )
                                elif diff_area is None:
                                    g.logger.debug(f"DEBUG>>>'MPD' {diff_area = } - whats the issue?")
                                else:
                                    g.logger.debug(f"WHATS GOING ON? {diff_area = } -- {max_diff_pixels = }")
                            # Saved does not intersect the current object/label
                            else:
                                g.logger.debug(
                                    f"{lp} current detection '{label[idx]}' is not near enough to '"
                                    f"{saved_ls[saved_idx]}' to evaluate for match past detection filter"
                                )
                                continue
                        if failed:
                            continue
                        # out of past detection bounding box loop, still inside if mpd

            elif (g.config.get("PAST_EVENT") and not str2bool(g.config.get("force_mpd"))) and (
                str2bool(mpd) or str2bool(seq_mpd)
            ):
                g.logger.debug(
                    f"{lp} this is a PAST event, skipping match past detections filter... override with "
                    f"'mpd_force=yes'"
                )
            elif (g.eid == saved_event) and (str2bool(mpd) or str2bool(seq_mpd)):
                g.logger.debug(
                    f"{lp} the current event is the same event as the last time this monitor processed an"
                    f" event, skipping match past detections filter... override with "
                    f"'mpd_force=yes'"
                )

            # end of main loop, if we made it this far label[idx] has passed filtering
            new_bbox.append(box[idx])
            new_label.append(label[idx])
            new_conf.append(conf[idx])
            new_model_names.append(model_names[idx])
            new_bbox_to_poly.append(b)
            appended = True
            g.logger.debug(2, f"+++ '{show_label}' has PASSED filtering")
            # failed = False
        # out of primary bounding box loop
        if failed:
            g.logger.debug(
                2,
                f"<<< '{label[-1]} ({tot_labels}/{tot_labels})' has FAILED filtering",
            )
            failed = False
        data = {
            "_b": new_bbox,
            "_l": new_label,
            "_c": new_conf,
            "_e": new_err,
            "_m": new_model_names,
        }
        extras = {
            "mpd_data": {
                "mpd_b": mpd_b,
                "mpd_c": mpd_c,
                "mpd_l": mpd_l,
            },
            "confidence_upper_break": conf_upper_break,
            "bbox_to_poly": {
                "new_bbox_to_poly": new_bbox_to_poly,
                "error_bbox_to_poly": error_bbox_to_poly,
            },
        }
        return data, extras

    # Run detection on a stream
    def detect_stream(self, stream, options=None, ml_overrides=None, in_file=False):
        def _pre_existing(pe_labels, labels_, origin):
            ret_val = False
            if pe_labels:
                g.logger.debug(2, f"pre_existing_labels: inside {origin}")
                if pe_labels == "pel_any" and not len(labels_):
                    # only run if this is the 1st sequence or there were no filtered
                    # detections after previous sequence
                    g.logger.debug(
                        f"pre existing labels: configured to 'pel_any' and there are not any detections "
                        f"as of yet, skipping model -> '{model_name}'"
                    )
                    ret_val = True
                elif pe_labels == "pel_none" and len(labels_):
                    g.logger.debug(
                        f"pre existing labels: configured to 'pel_none' and there are detections"
                        f", skipping model -> '{model_name}'"
                    )
                    ret_val = True
                elif not any(x in labels_ for x in pe_labels):
                    g.logger.debug(
                        f"pre_existing_labels: did not find {pe_labels} in {labels_}," f" skipping this model..."
                    )
                    ret_val = True
            return ret_val

        filtered_extras = {}
        if ml_overrides is None:
            ml_overrides = {}
        if options is None:
            options = {}
        saved_bs: list = []
        saved_ls: list = []
        saved_cs: list = []
        all_frames: list = []
        all_matches: list = []
        matched_b: list = []
        matched_e: list = []
        matched_l: list = []
        matched_c: list = []
        mpd_b: list = []
        mpd_c: list = []
        mpd_l: list = []
        matched_detection_types: list = []
        matched_frame_id: Optional[str] = None
        matched_model_names: list = []
        matched_frame_img: Optional[str] = None
        manual_locking: bool = False
        saved_event: Optional[str] = None
        pkl_data: dict = {}
        self.ml_overrides = ml_overrides
        self.stream_options = options
        # Hack to forward PAST_EVENT via stream options
        if not g.config.get("PAST_EVENT") and self.stream_options.get("PAST_EVENT"):
            g.config["PAST_EVENT"] = self.stream_options.get("PAST_EVENT")
        past_event: Optional[bool] = g.config.get("PAST_EVENT")

        frame_set: List[AnyStr] = self.stream_options.get("frame_set", ["snapshot", "alarm", "snapshot"])
        if frame_set and past_event:
            g.logger.debug(f"{lp} this is a past event, optimizing settings and workflow for speed")
            old_frame_set: list = frame_set
            new_frame_set: list = de_dup(frame_set)
            self.stream_options["frame_set"] = new_frame_set
            if len(new_frame_set) < len(old_frame_set):
                g.logger.debug(
                    f"{lp} optimized frame_set from {old_frame_set} -> {new_frame_set}",
                )
        frame_strategy: Optional[str] = self.stream_options.get("frame_strategy", "most_models")
        t: Timer = Timer()
        self.media = MediaStream(stream=stream, options=self.stream_options)
        polygons: Optional[list] = self.stream_options.get("polygons", [])
        if polygons:
            polygons = list(polygons)
        # todo: mpd as part of ml_overrides?
        mpd: Optional[Union[str, bool]] = self.ml_options.get("general", {}).get("match_past_detections")
        g.logger.debug(f"mpd:DBG>>> {self.ml_options.get('general', {}).get('match_past_detections') = } -- {mpd = }")
        mpd = str2bool(mpd)
        # Loops across all frames
        # match past detections is here, so we don't try and load/dump while still detecting
        # ml_overrides now has an enabled flag
        if str2bool(self.ml_overrides.get("enable")):
            g.logger.debug(f"{lp} ml_overrides are enabled! -> {self.ml_overrides}")
            self.model_sequence = self.ml_overrides.get("model_sequence").split(",")
            if self.ml_options.get("object", {}).get("general", {}).get("object_detection_pattern"):
                self.ml_options["object"]["general"]["object_detection_pattern"] = self.ml_overrides["object"][
                    "object_detection_pattern"
                ]
            if self.ml_options.get("face", {}).get("general", {}).get("face_detection_pattern"):
                self.ml_options["face"]["general"]["face_detection_pattern"] = self.ml_overrides["face"][
                    "face_detection_pattern"
                ]
            if self.ml_options.get("alpr", {}).get("general", {}).get("alpr_detection_pattern"):
                self.ml_options["alpr"]["general"]["alpr_detection_pattern"] = self.ml_overrides["alpr"][
                    "alpr_detection_pattern"
                ]

        if mpd:
            saved_bs, saved_ls, saved_cs, saved_event = pkl("load")
            pkl_data = {
                "saved_bs": saved_bs,
                "saved_ls": saved_ls,
                "saved_cs": saved_cs,
                "saved_event": saved_event,
                "mpd": mpd,
                "mpd_b": mpd_b,
                "mpd_c": mpd_c,
                "mpd_l": mpd_l,
            }
            g.logger.debug(
                f"mpd: last_event={saved_event} -- saved labels={saved_ls} -- saved_bbox={saved_bs} -- "
                f"saved conf={saved_cs}"
            )
        if len(self.model_sequence) > 1:
            g.logger.debug(
                2,
                f"{lp}portalocker: using automatic locking for switching between models",
            )
        else:
            manual_locking = True
            g.logger.debug(
                2,
                f"{lp}portalocker: using manual locking for single model -> '{self.model_sequence[0]}'",
            )
            # g.logger.debug(1,f"{self.ml_options=}")
            self.ml_options[self.model_sequence[0]]["auto_lock"] = False

        while self.media.more():
            frame: Optional[np.ndarray] = self.media.read()
            if frame is None:
                g.logger.debug(f"{lp} There are no more frames to process!")
                break
            # Start the timer for the current frame
            frame_perf_timer = Timer()
            _labels_in_frame: list = []
            _boxes_in_frame: list = []
            _error_boxes_in_frame: list = []
            _confs_in_frame: list = []
            _detection_types_in_frame: list = []
            _model_names_in_frame: list = []
            # remember this needs to occur after a frame
            # is read, otherwise we don't have dimensions
            if not polygons:
                polygons = []
                dimensions = self.media.image_dimensions()
                old_h = dimensions["original"][0]
                old_w = dimensions["original"][1]

                polygons.append(
                    {
                        "name": "full_image",
                        "value": [(0, 0), (old_w, 0), (old_w, old_h), (0, old_h)],
                        # 'value': [(0, 0), (5, 0), (5, 5), (0, 5)],
                        "pattern": None,
                    }
                )
                g.logger.debug(
                    2,
                    f"{lp} no polygons/zones specified, adding 'full_image' as polygon @ {polygons[0]['value']}",
                )

            if (not self.has_rescaled) and self.stream_options.get("resize", "no") != "no":
                self.has_rescaled = True
                dimensions = self.media.image_dimensions()
                old_h = dimensions["original"][0]
                old_w = dimensions["original"][1]
                new_h = dimensions["resized"][0]
                new_w = dimensions["resized"][1]
                if (old_h != new_h) or (old_w != new_w):
                    polygons[:] = self.rescale_polygons(polygons, new_w / old_w, new_h / old_h)
            # For each frame, loop across all models
            found = False
            if not isinstance(self.model_sequence, list):
                raise ValueError(f"No model sequences configured? FATAL!")
            for model_loop, model_name in enumerate(self.model_sequence):
                if str2bool(self.ml_overrides.get("enable")) and (
                    model_name not in self.ml_overrides.get("model_sequence")
                ):
                    g.logger.debug(f"{lp}overrides: '{model_name}' model is NOT in ml_overrides, skipping model...")
                    continue

                pre_existing_labels = self.ml_options.get(model_name, {}).get("general", {}).get("pre_existing_labels")
                if _pre_existing(pre_existing_labels, _labels_in_frame, f"{model_name}:general"):
                    continue

                # acquire BoundedSemaphore to control amount of models running at once
                if not self.models.get(model_name):
                    self._load_models([model_name])
                    if manual_locking:
                        for sequence in self.models[model_name]:
                            sequence.acquire_lock()
                same_model_sequence_strategy = (
                    self.ml_options.get(model_name, {}).get("general", {}).get("same_model_sequence_strategy", "most")
                )

                # start of same model iteration (sequences)
                _b_best_in_same_model: list = []
                _l_best_in_same_model: list = []
                _c_best_in_same_model: list = []
                _e_best_in_same_model: list = []
                _m_best_in_same_model: list = []
                _polygons_in_same_model: list = []
                _error_polygons_in_same_model: list = []
                _e: list = []
                # For each model, loop across different variations/sequences
                filtered: bool = False
                for sequence_loop, sequence in enumerate(self.models[model_name]):
                    seq_opt: dict = sequence.get_options()
                    # g.logger.debug(f"{seq_opt = }")
                    seq_mpd = seq_opt.get("match_past_detections")
                    if str2bool(seq_mpd):
                        g.logger.debug(
                            2,
                            f"mpd: '{sequence}' option match_past_detections configured",
                        )
                        if not len(saved_cs):
                            saved_bs, saved_ls, saved_cs, saved_event = pkl("load")
                    show_len: int = self.media.frame_set_len
                    if in_file and self.media.type == "file":
                        # --file image show_len = 1 ,video file and API both use frame_set
                        show_len = 1
                    g.logger.debug(
                        2,
                        f"frame: {self.media.last_frame_id_read} [strategy:'{frame_strategy}'] ("
                        f"{self.media.frames_processed} of {show_len}) - "
                        f"model: '{model_name}' [strategy:'{frame_strategy}'] ({model_loop + 1} of "
                        f"{len(self.model_sequence)}) - sequence: '{seq_opt['name']}' "
                        f"[strategy:'{same_model_sequence_strategy}'] ({sequence_loop + 1} of "
                        f"{len(self.models[model_name])})",
                    )
                    pre_existing_labels = seq_opt.get("pre_existing_labels")
                    if _pre_existing(
                        pre_existing_labels,
                        _labels_in_frame,
                        f"'ml_sequence':'{model_name}':'sequence':'{seq_opt['name']}'",
                    ):
                        continue

                    try:
                        _b, _l, _c, _m = sequence.detect(input_image=frame)
                    except Exception as e:
                        g.logger.error(f"{lp} error running sequence '{seq_opt['name']}'")
                        g.logger.error(f"{lp} ERR_MSG--> {e}")
                    else:
                        tot_labels = len(_l)
                        g.logger.debug(
                            2,
                            f"{lp} model: '{model_name}' seq: '{seq_opt['name']}' found {tot_labels} "
                            f"detection{'s' if tot_labels > 1 or tot_labels == 0 else ''}"
                            f" -> {', '.join(_l)}",
                        )
                        filtered_tot_labels: int = 0
                        # ONLY FILTER IF THERE ARE DETECTIONS
                        if tot_labels:
                            h, w = frame.shape[:2]
                            filtered_data, filtered_extras = self._filter_detections(
                                model_name,
                                _b,
                                _l,
                                _c,
                                polygons,
                                h,
                                w,
                                _m,
                                seq_opt=seq_opt,
                                model_type=model_name,
                                pkl_data=pkl_data,
                            )
                            _b = filtered_data["_b"]
                            _l = filtered_data["_l"]
                            _c = filtered_data["_c"]
                            _e = filtered_data["_e"]
                            _m = filtered_data["_m"]
                            filtered_tot_labels = len(_l)
                            _e_best_in_same_model.extend(_e)
                            if not filtered_tot_labels:
                                filtered = True

                        if filtered:
                            # reset to False, causes issues otherwise
                            filtered = False
                            continue
                        g.logger.debug(
                            2,
                            f"{lp}strategy: '{filtered_tot_labels}' filtered label"
                            f"{'s' if filtered_tot_labels > 1 else ''}: {_l} {_c} {_m} {_b}",
                        )
                        if filtered_tot_labels == 0:
                            filtered_extras["confidence_upper_break"] = False
                        # Moved here to be evaluated if changed by the 'confidence_upper' filter
                        added_ = False
                        if (
                            (same_model_sequence_strategy == "first")
                            or ((same_model_sequence_strategy == "most") and (len(_l) > len(_l_best_in_same_model)))
                            or (
                                (same_model_sequence_strategy == "most_unique")
                                and (len(set(_l)) > len(set(_l_best_in_same_model)))
                            )
                        ):
                            added_ = True
                            # self.has_rescaled = True
                            # dimensions = self.media.image_dimensions()
                            # old_h, old_w = self.media.orig_h_w
                            # new_h, new_w = self.media.resized_h_w
                            # if (old_h != new_h) or (old_w != new_w):
                            #     polygons[:] = self.rescale_polygons(
                            #         polygons, new_w / old_w, new_h / old_h
                            #     )
                            # if self.has_rescaled:
                            #     new_b = self.rescale_polygons()
                            _b_best_in_same_model = _b
                            _l_best_in_same_model = _l
                            _c_best_in_same_model = _c
                            _e_best_in_same_model = _e
                            _m_best_in_same_model = _m
                            # _polygons_in_same_model = bbox_to_polygon
                            # _error_polygons_in_same_model = error_bbox_to_polygon
                            # g.logger.debug(1,f"dbg:strategy: {_l_best_in_same_model = }")

                        elif same_model_sequence_strategy == "union":
                            added_ = True
                            _b_best_in_same_model.extend(_b)
                            _l_best_in_same_model.extend(_l)
                            _c_best_in_same_model.extend(_c)
                            _e_best_in_same_model.extend(_e)
                            _m_best_in_same_model.extend(_m)
                            # _polygons_in_same_model.extend(bbox_to_polygon)
                            # _error_polygons_in_same_model.extend(error_bbox_to_polygon)

                        if (same_model_sequence_strategy == "first") and len(_b):
                            g.logger.debug(
                                3,
                                f"{lp} breaking out of sequence loop as 'same_model_sequence_strategy' is first",
                            )
                            break
                        if (
                            same_model_sequence_strategy != "first"
                            and filtered_extras.get("confidence_upper_break") is True
                        ):
                            g.logger.debug(
                                3,
                                f"{lp}confidence_upper: short-circuiting 'same_model_sequence_strategy'",
                            )
                            if not added_ and len(_b):
                                g.logger.debug(
                                    f"DEBUG>>> this match was not added to best in same model yet, " f"ADDING NOW"
                                )
                                _b_best_in_same_model = _b
                                _l_best_in_same_model = _l
                                _c_best_in_same_model = _c
                                _e_best_in_same_model = _e
                                _m_best_in_same_model = _m
                            break
                # end of same model sequence iteration
                # at this state x_best_in_same_model contains the best match across
                # same model variations

                # still inside model loop
                if len(_l_best_in_same_model):
                    found = True
                    _labels_in_frame.extend(_l_best_in_same_model)
                    _boxes_in_frame.extend(_b_best_in_same_model)
                    _confs_in_frame.extend(_c_best_in_same_model)
                    _error_boxes_in_frame.extend(_e_best_in_same_model)
                    _detection_types_in_frame.extend([model_name] * len(_l_best_in_same_model))
                    _model_names_in_frame.extend(_m_best_in_same_model)
                    if same_model_sequence_strategy == "first":
                        g.logger.debug(
                            2,
                            f"{lp} breaking out of MODEL loop as 'same_model_sequence_strategy' is 'first'",
                        )
                        break
                else:
                    if not filtered and self.model_valid:
                        g.logger.debug(
                            2,
                            f"{lp} no '{model_name}' matches at all in frame: {self.media.last_frame_id_read}",
                        )
                    elif filtered and self.model_valid:
                        g.logger.debug(
                            2,
                            f"{lp} all '{model_name}' matches in frame {self.media.last_frame_id_read} "
                            f"were filtered out  -> {filtered = } --- {self.model_valid = }",
                        )
                    else:
                        g.logger.debug(
                            f"only 1 detection and it wasn't filtered out? -- IDK MAN come and check it out"
                            f" {filtered = } {self.model_valid = }"
                        )

            # end of model loop

            # still in frame loop
            frame_timer_end = frame_perf_timer.stop_and_get_ms()
            g.logger.debug(2, f"perf:frame: {self.media.last_frame_id_read} took {frame_timer_end}")
            if found:
                all_matches.append(
                    {
                        "labels": _labels_in_frame,
                        "model_names": _model_names_in_frame,
                        "confidences": _confs_in_frame,
                        "detection_types": _detection_types_in_frame,
                        "frame_id": self.media.last_frame_id_read,
                        "boxes": _boxes_in_frame,
                        "error_boxes": _error_boxes_in_frame,
                        "image": frame.copy(),
                    }
                )
                all_frames.append(self.media.get_last_read_frame())
                if frame_strategy == "first":
                    g.logger.debug(
                        2,
                        f"{lp} breaking out of frame loop as 'frame_strategy' is 'first'",
                    )
                    break
                elif filtered_extras.get("confidence_upper_break") is True:
                    g.logger.debug(3, f"{lp}confidence_upper: short-circuiting 'frame_strategy'")
                    break

        # end of while media loop
        # find best match in all_matches
        # matched_poly, matched_err_poly = [], []
        for idx, item in enumerate(all_matches):
            # g.logger.debug(1,f"dbg:strategy: most:[{len(item['labels']) = } > {len(matched_l) = }]
            # most_models:[{len(item['detection_types']) = } > {len(matched_detection_types) = }]"
            #                    f"most_unique:[{len(set(item['labels'])) = } > {len(set(matched_l)) = }]")
            # Credit to @hqhoang for this idea - https://github.com/ZoneMinder/pyzm/issues/36
            # Takes into consideration confidences instead of returning the first match.
            if (
                (frame_strategy == "first")
                or ((frame_strategy == "most") and (len(item["labels"]) > len(matched_l)))
                or (
                    (frame_strategy == "most")
                    and (len(item["labels"]) == len(matched_l))
                    and (sum(matched_c) < sum(item["confidences"]))
                )
                or ((frame_strategy == "most_models") and (len(item["detection_types"]) > len(matched_detection_types)))
                or (
                    (frame_strategy == "most_models")
                    and (len(item["detection_types"]) == len(matched_detection_types))
                    and (sum(matched_c) < sum(item["confidences"]))
                )
                or ((frame_strategy == "most_unique") and (len(set(item["labels"])) > len(set(matched_l))))
                or (
                    (frame_strategy == "most_unique")
                    and (len(set(item["labels"])) == len(set(matched_l)))
                    and (sum(matched_c) < sum(item["confidences"]))
                )
            ):
                # matched_poly = item['bbox2poly']
                matched_l = item["labels"]
                matched_model_names = item["model_names"]
                matched_c = item["confidences"]
                matched_frame_id = item["frame_id"]
                matched_detection_types = item["detection_types"]
                matched_b = item["boxes"]
                matched_e = item["error_boxes"]
                # matched_err_poly = item['err2poly']
                matched_frame_img = item["image"]
                # g.logger.debug(1,f"dbg:strategy: {matched_l = }")

        if manual_locking:
            for model_name in self.model_sequence:
                for sequence in self.models[model_name]:
                    sequence.release_lock()

        diff_time = t.stop_and_get_ms()
        matched_data = {
            "labels": matched_l,
            "model_names": matched_model_names,
            "confidences": matched_c,
            "frame_id": matched_frame_id,
            "type": matched_detection_types,
            "boxes": matched_b,
            # "bbox2poly": matched_poly,
            "image_dimensions": self.media.image_dimensions(),
            # 'stored_factors': (old_x_factor, old_y_factor),
            "polygons": polygons,
            "error_boxes": matched_e,
            # "err2poly": matched_err_poly,
            "image": matched_frame_img,
        }

        mon_name = f"'Monitor': {g.config.get('mon_name')} ({g.mid})->'Event': "
        g.logger.debug(
            f"perf:{lp}FINAL: {mon_name if isinstance(stream, int) else ''}"
            f"{stream} -> complete detection sequence took: {diff_time}"
        )
        if str2bool(mpd):
            # Write detections to the MPD buffer to be evaluated next time
            if filtered_extras:
                if matched_b and matched_b not in mpd_b:
                    mpd_b.extend(matched_b)
                    mpd_c.extend(matched_c)
                    mpd_l.extend(matched_l)
                    g.logger.debug(f"mpd: adding the current detection to the match_past_detection buffer!")
            pkl("write", mpd_b, mpd_l, mpd_c, g.eid)
        self.media.stop()
        # if invoked again, we need to resize polys
        self.has_rescaled = False
        return matched_data, all_matches, all_frames
