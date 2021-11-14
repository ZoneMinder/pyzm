"""
DetectSequence
=====================
Primary entry point to invoke machine learning classes in pyzm
It is recommended you only use DetectSequence methods and not
lower level interfaces as they may change drastically.
"""
# modified from source, original author @pliablepixels see: https://github.com/pliablepixels


import copy
import re
from typing import Optional

from shapely.geometry import Polygon
from traceback import format_exc

from pyzm.helpers.Media import MediaStream
from pyzm.helpers.pyzm_utils import (
    Timer,
    str2bool,
    pkl,
    de_dup,
    Pushover,
)

saved_bs, saved_ls, saved_cs, saved_event, mpd, eventid = (
    [],
    [],
    [],
    None,
    None,
    None,
)

mpd_b, mpd_l, mpd_c = [], [], []
PAST_EVENT: bool = False
g: Optional = None
lp: str = 'detect:'


class DetectSequence:
    def __init__(self, globs, options=None):
        """Initializes ML entry point with various parameters

        Args:
            - options (dict): Variety of ML options. Best described by an example as below

                ::

                    options = {
                        'general': {
                            # sequence of models you want to run for every specified frame
                            'model_sequence': 'object,face,alpr' ,
                            # If 'yes', will not use portalocks
                            'disable_locks':'no',

                        },

                        # We now specify all the config parameters per model_sequence entry
                        'object': {
                            'general':{
                                # 'first' - When detecting objects, if there are multiple fallbacks, break out
                                # the moment we get a match using any object detection library.
                                # 'most' - run through all libraries, select one that has most object matches
                                # 'most_unique' - run through all libraries, select one that has most unique object matches

                                'same_model_sequence_strategy': 'first' # 'first' 'most', 'most_unique', 'union'
                                'pattern': '.*' # any pattern
                            },

                            # within object, this is a sequence of object detection libraries. In this case,
                            # I want to first try on my TPU and if it fails, try GPU
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
                                'object_processor': 'gpu',
                                # These are optional below. Default is 416. Change if your model is trained for larger sizes
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
        g = globs
        if options is None:
            options = {}
        self.has_rescaled: bool = False  # only once in init
        self.model_sequence: Optional[str] = None
        self.ml_options: dict = {}
        self.stream_options: Optional[dict] = None
        self.media: Optional[MediaStream] = None
        self.ml_overrides: dict = {}
        self.models: dict = {}
        self.model_name: str = ""
        self.model_valid: bool = True
        self.raw_seq: dict = {}
        self.raw_frames: Optional[dict] = {}
        self.set_ml_options(options, force_reload=True)

    def get_ml_options(self):
        return self.ml_options

    def set_ml_options(self, options: dict, force_reload: bool = False):
        """Use this to change ml options later. Note that models will not be reloaded
        unless you add force_reload=True
        """
        # g.logger is not used because it is not initialized yet on first start
        model_sequence = options.get("general", {}).get('model_sequence', None)
        if isinstance(model_sequence, str):
            self.model_sequence = (
                options.get("general", {}).get("model_sequence", "DEFAULT").split(",")
            )

        else:
            self.model_sequence = options.get("general", {}).get(
                "model_sequence", ["DEFAULT2"]
            )
        self.model_sequence = [x.strip() for x in self.model_sequence]
        # print(f"{force_reload=} {preload=}")
        self.ml_options = options
        self.stream_options = None
        self.media = None
        self.ml_overrides = {}
        if force_reload:
            if self.models:
                self.models = {}

    def _load_models(self, models=None):
        def get_correct_model(frame_work):
            if frame_work == 'opencv':
                from pyzm.ml.yolo import Yolo
                return Yolo(options=model_sequence, globs=g)
            elif frame_work == 'coral_edgetpu':
                from pyzm.ml.coral_edgetpu import Tpu
                return Tpu(options=model_sequence, globs=g)
                # AWS Rekognition
            elif frame_work == 'aws_rekognition':
                from pyzm.ml.aws_rekognition import AwsRekognition
                return AwsRekognition(options=model_sequence, globs=g)
            else:
                raise ValueError(
                    f"Invalid object_framework: {frame_work}. Only opencv, coral_edgetpu "
                    f"(pycoral) and aws_rekognition supported as of now"
                )

        if not isinstance(models, list):
            g.logger.error(f"{lp} models must be a list of models (Yolo, Tpu, AWSRekognition)")
        if not models:
            if self.model_sequence and isinstance(self.model_sequence, str):
                models = self.model_sequence.split(',')
            else:
                models = self.model_sequence
        # print(f"****** {self.ml_options = }")
        import pyzm.ml.alpr as AlprDetect
        import pyzm.ml.face as FaceDetect
        accepted_models: tuple = ('object', 'face', 'alpr')
        for model in models:
            if model in accepted_models:
                self.models[model] = []
                sequences = self.ml_options.get(model, {}).get("sequence")
                if sequences:
                    for ndx, model_sequence in enumerate(sequences):
                        obj = None
                        seq_name = model_sequence.get('name')
                        if not seq_name:
                            seq_name = f"sequence:{ndx+1}"
                        if not str2bool(model_sequence.get("enabled", "yes")):
                            g.logger.debug(
                                2,
                                f"{lp} skipping sequence '{seq_name}' ({ndx + 1} of "
                                f"{len(sequences)}) as it is disabled"
                            )
                            continue
                        model_sequence["disable_locks"] = self.ml_options.get(
                            "general", {}).get("disable_locks", "no")
                        g.logger.debug(
                            2,
                            f"{lp} loading '{model}' sequence '{seq_name}' ({ndx + 1} of {len(sequences)}"
                        )
                        try:
                            if model == 'object':
                                framework = model_sequence.get('object_framework')
                                obj = get_correct_model(framework)
                            elif model == 'alpr':
                                obj = AlprDetect.Alpr(options=model_sequence, globs=g)
                            elif model == 'face':
                                obj = FaceDetect.Face(options=model_sequence, globs=g)
                            self.models[model].append(obj)
                        except Exception as exc:
                            g.logger.error(f"{lp} error while trying to construct '{model}' sequence "
                                           f"pipeline for '{seq_name}', skipping...")
                            g.logger.error(f"{lp} {exc}")
                else:
                    g.logger.debug(
                        f"{lp} '{model}' configured, but there are NO 'sequence' sections defined! skipping..."
                    )
            else:
                g.logger.error(f"{lp} the configured model '{model}' is not a valid type! skipping...")

    @staticmethod
    def rescale_polygons(polygons, x_factor, y_factor, method='down'):
        new_ps = []
        for p in polygons:
            # from pyzm.helpers.pyzm_utils import str2tuple
            # coords = str2tuple(p["value"])
            # p['value'] = coords
            newp = []
            for x, y in p["value"]:
                if method == 'down':
                    new_x = int(x * x_factor)
                    new_y = int(y * y_factor)
                elif method == 'up':
                    new_x = int(x / x_factor)
                    new_y = int(y / y_factor)
                else:
                    g.logger.error(f"{lp} rescale polygons -> can only use 'up' or 'down' for rescaling")
                    return
                newp.append((new_x, new_y))
            new_ps.append({
                "name": p["name"],
                "value": newp,
                "pattern": p["pattern"]
            })
        g.logger.debug(
            2,
            f"{lp} rescaled polygons ('{method}') using factors: x={x_factor:.4f}  y={y_factor:.4f}\n*** OLD:"
            f" [{polygons}] \n****** NEW:[{new_ps}]",
        )
        return new_ps

    def _filter_detections(
            self, seq, box, label, conf, polygons, h, w, model_names, seq_opt=None, model_type=None
    ):
        if seq_opt is None:
            seq_opt = {}
        moa, tot_labels, max_object_area, contained_area, failed, min_conf = (
            None,
            len(label),
            None,
            10.0,
            None,
            None,
        )
        global mpd, mpd_b, mpd_c, mpd_l
        lp: str = 'detect:filtering:'
        # g.logger.debug(4, f"|--- Filtering {tot_labels} {'detections' if tot_labels > 1 else 'detection'} ---|")
        new_label, new_bbox, new_conf, new_err, new_model_names = [], [], [], [], []
        new_bbox_to_poly, error_bbox_to_poly = [], []
        min_conf_found = ""
        ioa_found = ""
        ioa = None
        moa = None
        moa_found = ""
        if seq_opt.get("object_min_confidence"):
            min_conf = seq_opt.get("object_min_confidence")
            min_conf_found = f"object_min_conf:sequence->{seq_opt.get('name')}"
        if seq_opt.get("max_detection_size"):
            moa = seq_opt.get("max_detection_size")
            moa_found = f"max_detection_size:sequence->{seq_opt.get('name')}"
        if self.ml_options.get("object", {}).get("general", {}).get("contained_area"):
            ioa = (
                self.ml_options.get("object", {})
                    .get("general", {})
                    .get("contained_area")
            )
            ioa_found = f"contained_area:object->general"
        if seq_opt.get("contained_area"):
            ioa = seq_opt.get("contained_area")
            ioa_found = f"contained_area:sequence->{seq_opt.get('name')}"

        # g.logger.debug(f"before looping {g.config.get('contained_area') = } {ioa = } {ioa_found = } {min_conf = } {min_conf_found = } {moa = } {moa_found = }")
        for idx, b in enumerate(box):
            if failed:
                g.logger.debug(
                    2,
                    f"detection: '{label[idx - 1]} ({idx}/{tot_labels})' has FAILED filtering",
                )  # for each object that failed before loop end
                failed = None
            show_label = f"{label[idx]} ({idx + 1}/{tot_labels})"
            g.logger.debug(
                f">>> detected '{show_label}' confidence: {conf[idx]:.2f}"
            )
            old_b = b
            it = iter(b)
            b = list(zip(it, it))
            b.insert(1, (b[1][0], b[0][1]))
            b.insert(3, (b[0][0], b[2][1]))
            # save b as the objects polygon
            obj = Polygon(b)
            # get minimum confidence override per label
            if self.ml_options.get("general", {}).get(
                    f"{label[idx]}_min_confidence"
            ):
                min_conf = self.ml_options.get("general", {}).get(
                    f"{label[idx]}_min_confidence"
                )
                min_conf_found = "overridden:ml_sequence->general"
            if self.ml_options.get("general", {}).get(
                    f"{label[idx]}_min_confidence"
            ):
                min_conf = self.ml_options.get("general", {}).get(
                    f"{label[idx]}_min_confidence"
                )
                min_conf_found = "global config->ml_sequence->general"

            # get intersection area of bounding box inside polygon zone
            if self.ml_options.get("general", {}).get(f"{label[idx]}_contained_area"):
                ioa = self.ml_options.get("general", {}).get(
                    f"{label[idx]}_contained_area"
                )
                ioa_found = "overriden:ml_sequence->general"

            # max detected object area
            if self.ml_options.get("general", {}).get(
                    f"{label[idx]}_max_detection_size"
            ):
                moa = self.ml_options.get("general", {}).get(
                    f"{label[idx]}_max_detection_size"
                )
                moa_found = "overriden:ml_sequence->general"

            # do confidence filtering first then max object area
            try:
                min_conf = float(min_conf)
            except Exception:
                min_conf = 0.5
            if min_conf:
                g.logger.debug(
                    f"'{show_label}' minimum confidence found: ({min_conf_found}) -> '{min_conf}'"
                )
                if conf[idx] < min_conf:  # confidence filter
                    g.logger.debug(
                        2,
                        f"confidence: {conf[idx] * 100:.2f} is lower than minimum of {min_conf * 100:.2f}, removing..."
                    )
                    # don't draw red error boxes around filtered out objects by confidence if not specified in config
                    if str2bool(g.config.get("show_conf_filtered")):
                        error_bbox_to_poly.append(b)
                        new_err.append(old_b)
                    failed = True
                    continue

            if moa and moa != "100%":
                g.logger.debug(
                    4,
                    f"'{show_label}' max area of detected object found ({moa_found}) -> '{moa}'",
                )
                # Let's make sure its the right size
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
                        f"max size: {obj.area:.2f} is larger then the max allowed: {max_object_area:.2f}, removing...",
                    )
                    failed = True
                    error_bbox_to_poly.append(b)
                    new_err.append(old_b)
                    continue

            pattern_match = None
            for p in polygons:  # are there more than 1 polygon/zone masks to compare to?
                polygon_zone = Polygon(p['value'])
                g.logger.debug(
                    2,
                    f"checking if '{show_label}' @ {b} is inside polygon/zone '{p['name']}' located at {p['value']}",
                )
                if obj.intersects(polygon_zone):
                    g.logger.debug(
                        f"'{show_label}' INTERSECTS polygon/zone '{p['name']}'"
                    )
                    pixels_inside = obj.intersection(polygon_zone).area
                    percent_inside = (pixels_inside / obj.area) * 100
                    if p["name"] != "full_image":
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
                            # Let's make sure its the right size
                            _m = re.match(r"(\d*\.?\d*)(px|%)?$", ioa, re.IGNORECASE)
                            if _m:
                                contained_area = float(_m.group(1))
                                if _m.group(2) == "%":
                                    contained_area = (
                                            float(_m.group(1)) / 100.0 * obj.area
                                    )
                                    g.logger.debug(
                                        2,
                                        f"contained within zone: converted {_m.group(1)}% of {show_label}'s area "
                                        f"({obj.area}) to {contained_area} pixels",
                                    )
                        # setup a contained within polygon filter here
                        if pixels_inside < contained_area:
                            g.logger.debug(
                                f"'{show_label}' only has {pixels_inside:.2f} pixels ({percent_inside:.2f}%) inside "
                                f"of {p.get('name')} minimum is -> {contained_area} ("
                                f"{round((contained_area / obj.area) * 100, 2)}%)"
                            )
                            error_bbox_to_poly.append(b)
                            new_err.append(old_b)
                            failed = True
                            continue
                    # pattern matching is here because polygon/zone might have its own match pattern
                    if str2bool(self.ml_overrides.get('enable')) and self.ml_overrides.get(seq, {}).get(
                            '{}_detection_pattern'.format(model_type)):
                        match_pattern = self.ml_overrides.get(seq, {}).get(
                            '{}_detection_pattern'.format(model_type))
                        g.logger.debug(2, "match pattern: overridden by ml_overrides from '{}' to '{}'".format(
                            self.ml_options.get(seq, {}).get('general', {}).get(
                                '{}_detection_pattern'.format(model_type), '.*'), match_pattern))
                    elif p["pattern"]:
                        g.logger.debug(
                            3,
                            "detection label match pattern: "
                            "zone '{}' has overrides->'{}'".format(
                                p["name"],
                                p["pattern"],
                                self.ml_options.get(seq, {})
                                    .get("general", {})
                                    .get("pattern", ".*"),
                            ),
                        )
                        match_pattern = p["pattern"]
                    else:
                        match_pattern = (
                            self.ml_options.get(seq, {})
                                .get("general", {})
                                .get('{}_detection_pattern'.format(model_type), ".*")
                        )
                    g.logger.debug(2, f"match pattern: {match_pattern}")

                    r = re.compile(match_pattern)
                    match = list(filter(r.match, label))

                    if label[idx] not in match:
                        error_bbox_to_poly.append(b)
                        new_err.append(old_b)
                        g.logger.debug(
                            3,
                            f"match pattern: '{show_label}' does not match pattern filter, removing...",
                        )
                        continue
                    elif label[idx] in match:
                        pattern_match = True

                else:
                    error_bbox_to_poly.append(b)
                    new_err.append(old_b)
                    g.logger.debug(
                        2,
                        f"intersection: '{show_label}' does not intersect zone: {p['name']}, removing...",
                    )
            # out of polygon/zone loop
            if not pattern_match:
                failed = True
                continue

            # MATCH PAST DETECTIONS
            # todo add buffer and time based configurations
            seq_mpd = seq_opt.get("match_past_detections")
            # g.logger.debug(f"{type(saved_event)=} {type(eventid)=}")
            if (str2bool(mpd) or str2bool(seq_mpd)) and (
                    not eventid == saved_event
                    or (
                            not g.config.get("PAST_EVENT")
                            or g.config.get("PAST_EVENT")
                            and g.config.get("mpd_force")
                    )
            ):
                if not saved_bs:
                    g.logger.debug(
                        f"mpd: there are no saved detections to evaluate, skipping match past detection filter"
                    )
                else:
                    (
                        mda_found,
                        max_diff_area,
                        use_percent,
                        ignore_mpd,
                        removed_by_mpd,
                        mda,
                    ) = ("", None, None, None, None, None)
                    if self.ml_options.get("general", {}).get(
                            f"past_det_max_diff_area"
                    ):
                        mda = self.ml_options.get("general", {}).get(
                            f"past_det_max_diff_area"
                        )
                        mda_found = "past_det_max_diff_area:general"
                    if self.ml_options.get("general", {}).get(
                            f"{label[idx]}_past_det_max_diff_area"
                    ):
                        mda = self.ml_options.get("general", {}).get(
                            f"{label[idx]}_past_det_max_diff_area"
                        )
                        mda_found = "overriden:general"
                    mpd_ig = seq_opt.get(
                        "ignore_past_detection_labels",
                        self.ml_options.get("general", {}).get(
                            "ignore_past_detection_labels", []
                        ),
                    )

                    if seq_opt.get("past_det_max_diff_area"):
                        mda = seq_opt.get("past_det_max_diff_area")
                        mda_found = (
                            f'past_det_max_diff_area:sequence->{seq_opt.get("name")}'
                        )
                    if seq_opt.get(f"{label[idx]}_past_det_max_diff_area"):
                        mda = seq_opt.get(f"{label[idx]}_past_det_max_diff_area")
                        mda_found = f'overriden:sequence->{seq_opt.get("name")}'

                    if mpd_ig and label[idx] in mpd_ig:
                        g.logger.debug(
                            4,
                            "mpd: {} is in ignore list: {}, skipping".format(
                                label[idx], mpd_ig
                            ),
                        )
                        ignore_mpd = True
                    if mda and isinstance(mda, str) and str(mda).startswith("0"):
                        g.logger.debug(
                            4,
                            "mpd:  is set to {} (Leading 0 means BYPASS), skipping".format(
                                label[idx], mda
                            ),
                        )
                        ignore_mpd = True
                    if ignore_mpd:  # continue means it wont be removed
                        new_bbox.append(box[idx])
                        new_label.append(label[idx])
                        new_conf.append(conf[idx])
                        new_model_names.append(model_names[idx])
                        new_bbox_to_poly.append(b)
                        continue

                    if mda:
                        _m = re.match(r"(\d+)(px|%)?$", mda, re.IGNORECASE)
                        if _m:
                            max_diff_area = float(_m.group(1))
                            use_percent = (
                                True
                                if _m.group(2) is None or _m.group(2) == "%"
                                else False
                            )
                        else:
                            g.logger.error(
                                f"mpd:  malformed -> {mda}, setting to 5%..."
                            )
                            use_percent = True
                            max_diff_area = 5.0

                        # it's very easy to forget to add 'px' when using pixels
                        if use_percent and (max_diff_area < 0 or max_diff_area > 100):
                            g.logger.error(
                                "mpd: {} must be in the range 0-100 when using percentages: {}, setting to 5%...".format(
                                    max_diff_area, mda
                                )
                            )
                            max_diff_area = 5.0
                    else:
                        g.logger.debug(
                            f"mpd: no past_det_max_diff_area or per label overrides configured while "
                            f"match_past_detections=yes, setting to 5% as default"
                        )
                        max_diff_area = 5.0
                        use_percent = True

                    g.logger.debug(
                        4,
                        "mpd: max difference in area configured ({}) -> '{}', comparing past detections to "
                        "current".format(mda_found, mda),
                    )

                    for saved_idx, saved_b in enumerate(saved_bs):
                        # compare current detection element with saved list from file
                        found_past_match = False  #
                        aliases = self.ml_options.get("general", {}).get("aliases", {})
                        if saved_ls[saved_idx] != label[idx]:
                            if aliases and isinstance(aliases, dict):
                                g.logger.debug(f"mpd: checking aliases")
                                for item, value in (
                                        self.ml_options.get("general", {})
                                        .get("aliases", {})
                                        .items()
                                ):
                                    if found_past_match:
                                        break
                                    elif (
                                            saved_ls[saved_idx] in value
                                            and label[idx] in value
                                    ):
                                        found_past_match = True
                                        g.logger.debug(
                                            2,
                                            "mpd:aliases: found current label -> '{}' and past label -> '{}' "
                                            "are in an alias group named -> '{}'".format(
                                                label[idx], saved_ls[saved_idx], item
                                            ),
                                        )
                            elif aliases and not isinstance(aliases, dict):
                                g.logger.debug(
                                    f"mpd: aliases are configured but the format is incorrect, check the example "
                                    f"config for formatting and reformat aliases to a dictionary type setup"
                                )

                        elif saved_ls[saved_idx] == label[idx]:
                            found_past_match = True
                        if not found_past_match:
                            continue
                        it = iter(saved_b)
                        saved_b = list(zip(it, it))
                        saved_b.insert(1, (saved_b[1][0], saved_b[0][1]))
                        saved_b.insert(3, (saved_b[0][0], saved_b[2][1]))
                        saved_obj = Polygon(saved_b)
                        max_diff_pixels = None
                        g.logger.debug(
                            4,
                            "mpd: comparing '{}' PAST->{} to CURR->{}".format(
                                label[idx], saved_b, b
                            ),
                        )
                        if saved_obj.intersects(obj):
                            g.logger.debug(
                                4, f"mpd: the past object INTERSECTS the new object"
                            )
                            if obj.contains(saved_obj):
                                diff_area = obj.difference(saved_obj).area
                                if use_percent:
                                    max_diff_pixels = obj.area * max_diff_area / 100
                            else:
                                diff_area = saved_obj.difference(obj).area
                                if use_percent:
                                    max_diff_pixels = (
                                            saved_obj.area * max_diff_area / 100
                                    )
                            if diff_area <= max_diff_pixels:
                                g.logger.debug(
                                    "mpd: removing '{}' as it seems to be in the same spot as it was detected "
                                    "last time based on '{}' -> NOW: {} --- PAST: {}".format(
                                        show_label, mda, b, saved_b
                                    )
                                )
                                removed_by_mpd = True
                            else:
                                g.logger.debug(
                                    4,
                                    "mpd: allowing '{}' -> the difference in the area of last detection to this "
                                    "detection is '{:.2f}', a minimum of {:.2f} is needed to not be considered "
                                    "'in the same spot'".format(
                                        show_label, diff_area, max_diff_pixels
                                    ),
                                )
                        elif obj.intersects(saved_obj):
                            g.logger.debug(
                                4, f"mpd: the NEW object INTERSECTS the past object"
                            )
                            if obj.contains(saved_obj):
                                diff_area = obj.difference(saved_obj).area
                                if use_percent:
                                    max_diff_pixels = obj.area * max_diff_area / 100
                            else:
                                diff_area = saved_obj.difference(obj).area
                                if use_percent:
                                    max_diff_pixels = (
                                            saved_obj.area * max_diff_area / 100
                                    )
                            if diff_area <= max_diff_pixels:
                                g.logger.debug(
                                    "mpd: removing '{}' as it seems to be in the same spot as it was detected "
                                    "last time -> NOW: {} -- PAST: {}".format(
                                        show_label, b, saved_b
                                    )
                                )
                                removed_by_mpd = True
                            elif diff_area == 0:
                                g.logger.debug(
                                    "mpd: removing '{}' as it is in the EXACT same spot as it was detected "
                                    "last time -> NOW: {} -- PAST: {}".format(
                                        show_label, b, saved_b
                                    )
                                )
                                removed_by_mpd = True
                            else:
                                g.logger.debug(
                                    4,
                                    "mpd: allowing '{}' -> the difference in the area of last detection to this "
                                    "detection is '{:.2f}', a minimum of {:.2f} is needed to not be considered "
                                    "'in the same spot'".format(
                                        show_label, diff_area, max_diff_pixels
                                    ),
                                )
                        else:  # no where near each other
                            g.logger.debug(
                                f"mpd: current detection '{label[idx]}' is not near enough to '"
                                f"{saved_ls[saved_idx]}' to evaluate for match past detection filter"
                            )
                            continue
                        if removed_by_mpd:
                            if (
                                    saved_bs[saved_idx] not in mpd_b
                                    and saved_cs[saved_idx] not in mpd_c
                            ):
                                g.logger.debug(
                                    f"mpd: saving the removed detection to re-add to the buffer for next detection"
                                )
                                mpd_b.append(saved_bs[saved_idx])
                                mpd_c.append(saved_cs[saved_idx])
                                mpd_l.append(saved_ls[saved_idx])
                            new_err.append(
                                b
                            )  # b is Polygon ready, box[idx] is the top left(x,y), bottom right(x,y)
                            continue
                    # out of past detection bounding box loop, still inside if mpd
                    if removed_by_mpd:
                        failed = True
                        continue

            elif (g.config.get("PAST_EVENT")) and (str2bool(mpd) or str2bool(seq_mpd)):
                g.logger.debug(
                    f"mpd: this is a PAST event, skipping match past detections filter... override with "
                    f"'mpd_force=yes'"
                )
            elif str(eventid) == saved_event and (str2bool(mpd) or str2bool(seq_mpd)):
                g.logger.debug(
                    f"mpd: the current event is the same event as the last time this monitor processed an"
                    f" event, skipping match past detections filter"
                )

            # end of main loop, if we made it this far label[idx] has passed filtering
            new_bbox.append(box[idx])
            new_label.append(label[idx])
            new_conf.append(conf[idx])
            new_model_names.append(model_names[idx])
            new_bbox_to_poly.append(b)
            g.logger.debug(2, f"detection: '{show_label}' has PASSED filtering")
        # out of primary bounding box loop
        if failed:
            g.logger.debug(
                2,
                f"detection: '{label[-1]} ({tot_labels}/{tot_labels})' has FAILED filtering",
            )

        return new_bbox, new_label, new_conf, new_err, new_model_names, new_bbox_to_poly, error_bbox_to_poly

    # Run detection on a stream
    def detect_stream(self, stream, options=None, ml_overrides=None, sub_options=None, in_file=False):
        global saved_bs, saved_ls, saved_cs, saved_event, mpd, PAST_EVENT, eventid
        if ml_overrides is None:
            ml_overrides = {}
        if options is None:
            options = {}
        (
            saved_bs,
            saved_ls,
            saved_cs,
            all_frames,
            all_matches,
            matched_b,
            matched_e,
            matched_l,
            matched_c,
            matched_detection_types,
            matched_frame_id,
            matched_images,
            matched_model_names,
            matched_frame_img,
            manual_locking,
            saved_event,
        ) = ([], [], [], [], [], [], [], [], [], [], None, [], [], None, False, None)
        eventid = stream
        self.ml_overrides = ml_overrides
        self.stream_options = options
        if not g.config.get("PAST_EVENT") and self.stream_options.get("PAST_EVENT"):
            g.config["PAST_EVENT"] = self.stream_options.get("PAST_EVENT")
        frame_set = self.stream_options.get("frame_set", ["snapshot", "alarm", "snapshot"])
        PAST_EVENT = g.config.get("PAST_EVENT", self.stream_options.get("PAST_EVENT"))
        if frame_set and PAST_EVENT:
            g.logger.debug(
                f"{lp} this is a past event, optimizing settings and workflow for speed"
            )
            old_frame_set = frame_set
            new_frame_set = de_dup(frame_set)
            self.stream_options["frame_set"] = new_frame_set
            if len(new_frame_set) < len(old_frame_set):
                g.logger.debug(
                    f"{lp} optimized frame_set from {old_frame_set} -> {new_frame_set}",
                )
        frame_strategy = self.stream_options.get("frame_strategy", "most_models")
        # g.logger.debug(1,f"{lp} provided stream_sequence = {self.stream_options}")
        t = Timer()
        self.media = MediaStream(stream, "video", self.stream_options, g)
        polygons = self.stream_options.get("polygons", [])
        if polygons:
            polygons = list(polygons)
        # todo: mpd as part of ml_overrides?
        mpd = self.ml_options.get("general", {}).get("match_past_detections")
        # Loops across all frames
        # match past detections is here so we don't try and load/dump while still detecting
        # todo: ADD SMART BUFFER TO MPD - with timeout
        # g.logger.debug(f"{self.ml_options = }")
        if str2bool(self.ml_overrides.get('enable')):
            g.logger.debug(f"{lp} ml_overrides are enabled! -> {self.ml_overrides}")
            self.model_sequence = (self.ml_overrides.get("model_sequence").split(","))
            if self.ml_options.get('object', {}).get('general', {}).get('object_detection_pattern'):
                self.ml_options['object']['general']['object_detection_pattern'] = self.ml_overrides['object']['object_detection_pattern']
            if self.ml_options.get('face', {}).get('general', {}).get('face_detection_pattern'):
                self.ml_options['face']['general']['face_detection_pattern'] = self.ml_overrides['face']['face_detection_pattern']
            if self.ml_options.get('alpr', {}).get('general', {}).get('alpr_detection_pattern'):
                self.ml_options['alpr']['general']['alpr_detection_pattern'] = self.ml_overrides['alpr']['alpr_detection_pattern']

        if str2bool(mpd):
            g.logger.debug(
                2,
                f"{lp}mpd:  "
                f"mpd={self.ml_options.get('general', {}).get('match_past_detections', 'N/A')} ",
            )
            saved_bs, saved_ls, saved_cs, saved_event = pkl("load")
            g.logger.debug(f"mpd: last_event={saved_event} -- saved labels=[{saved_ls}] -- saved_bbox=[{saved_bs}] -- "
                           f"saved conf=[{saved_cs}]")
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
        # turn to a for loop by adding an __iter__
        while self.media.more():
            # for frame in self.media.more(past_event=past_event):
            frame = self.media.read()
            if frame is None:
                g.logger.debug(
                    f"{lp} The media object has stopped supplying frames!"
                )
                break
            # Start the timer for the current frame
            frame_perf_timer = Timer()
            (
                _labels_in_frame,
                _boxes_in_frame,
                _error_boxes_in_frame,
                _confs_in_frame,
                _detection_types_in_frame,
                _model_names_in_frame,
            ) = ([], [], [], [], [], [])
            # remember this needs to occur after a frame
            # is read, otherwise we don't have dimensions
            # if self.media.frames_processed == 1:  # should only need to do this once
            if not polygons:
                polygons = []
                old_h = self.media.image_dimensions()["original"][0]
                old_w = self.media.image_dimensions()["original"][1]

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
                    f"{lp} no polygons/zones specified, adding 'full_image' as polygon @ {polygons[0]['value']}"
                )

            if (not self.has_rescaled) and self.stream_options.get(
                    "resize", "no"
            ) != "no":
                self.has_rescaled = True
                dimensions = self.media.image_dimensions()
                old_h = dimensions["original"][0]
                old_w = dimensions["original"][1]
                new_h = dimensions["resized"][0]
                new_w = dimensions["resized"][1]
                if (old_h != new_h) or (old_w != new_w):
                    polygons[:] = self.rescale_polygons(
                        polygons, new_w / old_w, new_h / old_h
                    )

            # For each frame, loop across all models
            found = False
            if not isinstance(self.model_sequence, list):
                raise ValueError(f"No model sequences configured? FATAL!")
            for model_loop, model_name in enumerate(self.model_sequence):
                if (
                        str2bool(self.ml_overrides.get('enable'))
                        and
                        (model_name not in self.ml_overrides.get('model_sequence'))
                ):
                    g.logger.debug(
                        f"{lp}overrides: '{model_name}' model is NOT in ml_overrides, skipping model...")
                    continue
                pre_existing_labels = (
                    self.ml_options.get(model_name, {})
                        .get("general", {})
                        .get("pre_existing_labels")
                )
                if pre_existing_labels:
                    g.logger.debug(
                        2, f"pre_existing_labels: inside '{model_name}':'general' "
                    )
                    if pre_existing_labels == "pel_any" and not len(_labels_in_frame):
                        # only run if this is the 1st sequence or there were no filtered
                        # detections after previous sequence
                        g.logger.debug(
                            f"pre existing labels: configured to 'pel_any' and there are not any detections "
                            f"as of yet, skipping model -> '{model_name}'"
                        )
                        continue
                    elif pre_existing_labels == "pel_none" and len(_labels_in_frame):
                        g.logger.debug(
                            f"pre existing labels: configured to 'pel_none' and there are detections"
                            f", skipping model -> '{model_name}'"
                        )
                        continue
                    elif not any(x in _labels_in_frame for x in pre_existing_labels):
                        g.logger.debug(
                            f"pre_existing_labels: did not find {pre_existing_labels} in {_labels_in_frame},"
                            f" skipping this model..."
                        )
                        continue
                if not self.models.get(model_name):
                    self._load_models([model_name])
                    if manual_locking:
                        for sequence in self.models[model_name]:
                            sequence.acquire_lock()
                same_model_sequence_strategy = (
                    self.ml_options.get(model_name, {})
                        .get("general", {})
                        .get("same_model_sequence_strategy", "most")
                )
                # start of same model iteration
                (
                    _b_best_in_same_model,
                    _l_best_in_same_model,
                    _c_best_in_same_model,
                    _e_best_in_same_model,
                    _m_best_in_same_model,
                ) = ([], [], [], [], [])
                _polygons_in_same_model, _error_polygons_in_same_model = [], []

                _e = None
                # For each model, loop across different variations/sequences
                filtered = False
                for sequence_loop, sequence in enumerate(self.models[model_name]):
                    seq_opt = sequence.get_options()
                    # g.logger.debug(f"{seq_opt = }")
                    seq_mpd = seq_opt.get("match_past_detections")
                    if str2bool(seq_mpd):
                        g.logger.debug(
                            2,
                            f"mpd: '{sequence}' option match_past_detections configured",
                        )
                        if not len(saved_cs):  # todo add a global variable to let me know if it has been loaded yet
                            saved_bs, saved_ls, saved_cs, saved_event = pkl("load")
                    show_len = len(self.media.frame_set)
                    # g.logger.debug(f"{in_file=} {self.media.type=}")
                    if in_file and self.media.type == 'file':
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
                    if pre_existing_labels:
                        g.logger.debug(
                            2,
                            f"pre_existing_labels: inside 'ml_sequence':'{model_name}':'sequence':'{seq_opt['name']}'",
                        )
                        if pre_existing_labels == "pel_any" and not len(
                                _l_best_in_same_model
                        ):
                            # only run if this is the 1st sequence or there were no filtered detections after previous sequence
                            g.logger.debug(
                                f"pre existing labels: configured to 'pel_any' and there are not any detections "
                                f"as of yet, SKIPPING model {model_name} sequence -> '{seq_opt['name']}'"
                            )
                            continue
                        elif pre_existing_labels == "pel_none" and len(
                                _l_best_in_same_model
                        ):
                            g.logger.debug(
                                f"pre existing labels: configured to 'pel_none' and there are detections"
                                f", SKIPPING model {model_name} sequence -> '{seq_opt['name']}'"
                            )
                            continue
                        elif not any(
                                x in _l_best_in_same_model for x in pre_existing_labels
                        ):
                            g.logger.debug(
                                f"pre_existing_labels: did not find configured {pre_existing_labels} in "
                                f"detected -> {_l_best_in_same_model}, skipping this model sequence...",
                            )
                            continue

                    tot_labels = None

                    # bbox_to_polygon = None
                    # error_bbox_to_polygon = None
                    try:
                        # Keep a copy of the images in memory, return the image in a response if this is a mlapi request
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
                        filtered_tot_labels = None
                        if tot_labels:  # ONLY FILTER IF THERE ARE DETECTIONS
                            h, w = frame.shape[:2]
                            _b, _l, _c, _e, _m, bbox_to_polygon, error_bbox_to_polygon = self._filter_detections(
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

                            )
                            filtered_tot_labels = len(_l)
                            if _e is not None:
                                _e_best_in_same_model.extend(_e)
                            if not filtered_tot_labels:
                                filtered = True  # nothing left after filtering

                        if not filtered_tot_labels:
                            continue

                        # g.logger.debug(1,f"dbg:strategy: most:[if {len(_l) = } > {len(_l_best_in_same_model) = }]"
                        #                    f"most_unique:[if {len(set(_l)) = } > {len(set(_l_best_in_same_model)) = }]")
                        g.logger.debug(
                            2,
                            f"{lp}strategy: '{filtered_tot_labels}' filtered label"
                            f"{'s' if filtered_tot_labels > 1 else ''}: {_l} {_c} {_m} {_b}",
                        )
                        high_conf = str2bool(g.config.get("same_model_high_conf"))
                        if (
                                (same_model_sequence_strategy == "first")
                                or (
                                (same_model_sequence_strategy == "most")
                                and (len(_l) > len(_l_best_in_same_model))
                                )
                                or (
                                (same_model_sequence_strategy == "most_unique")
                                and (len(set(_l)) > len(set(_l_best_in_same_model)))
                                )
                        ):
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
                            _b_best_in_same_model.extend(_b)
                            _l_best_in_same_model.extend(_l)
                            _c_best_in_same_model.extend(_c)
                            _e_best_in_same_model.extend(_e)
                            _m_best_in_same_model.extend(_m)
                            # _polygons_in_same_model.extend(bbox_to_polygon)
                            # _error_polygons_in_same_model.extend(error_bbox_to_polygon)

                        if (
                                high_conf
                        ):
                            # g.logger.debug(
                            #     f"HIGH_CONF=YES: current best detection {repr(_l_best_in_same_model)}->{repr(_c_best_in_same_model)}"
                            #     f"-->{repr(_m_best_in_same_model)} --- current comparison "
                            #     f"{repr(_l)}->{repr(_c)}-->{repr(_m)}"
                            # )
                            # detection but different models and pick the higher confidence match. Currently it will add
                            # duplicates which causes unnecessary clutter in image and detection data
                            # todo: clean this up, POC works
                            nb_l, nb_c, nb_b, nb_e, nb_m = [], [], [], [], []
                            if len(_b_best_in_same_model):
                                max_diff_pixels = None
                                max_diff_area = 15.00  # configurable?
                                use_percent = True
                                for idx, best_bbox in enumerate(_b_best_in_same_model):
                                    it = iter(best_bbox)
                                    poly_bbox = list(zip(it, it))
                                    poly_bbox.insert(
                                        1, (poly_bbox[1][0], poly_bbox[0][1])
                                    )
                                    poly_bbox.insert(
                                        3, (poly_bbox[0][0], poly_bbox[2][1])
                                    )
                                    best_obj = Polygon(poly_bbox)

                                    for new_idx, new_bbox in enumerate(_b):
                                        if _l_best_in_same_model[idx] != _l[new_idx]:
                                            continue  # aliases?
                                        if (
                                                _c_best_in_same_model[idx] == _c[new_idx]
                                                and _b_best_in_same_model[idx]
                                                == _b[new_idx]
                                        ):
                                            # g.logger.debug(
                                            #     f"high confidence match: skipping this comparison as the two "
                                            #     f"detections that are about to be compared are the exact same"
                                            # )
                                            continue
                                        new_it = iter(new_bbox)
                                        npoly_bbox = list(zip(new_it, new_it))
                                        npoly_bbox.insert(
                                            1, (npoly_bbox[1][0], npoly_bbox[0][1])
                                        )
                                        npoly_bbox.insert(
                                            3, (npoly_bbox[0][0], npoly_bbox[2][1])
                                        )
                                        new_obj = Polygon(npoly_bbox)
                                        if new_obj.intersects(best_obj):
                                            # g.logger.debug(
                                            #     4,
                                            #     f"high confidence match: the current object INTERSECTS"
                                            #     f" the best object",
                                            # )
                                            if new_obj.contains(best_obj):
                                                diff_area = new_obj.difference(
                                                    best_obj
                                                ).area
                                                if use_percent:
                                                    max_diff_pixels = (
                                                            new_obj.area
                                                            * max_diff_area
                                                            / 100
                                                    )
                                            else:
                                                diff_area = best_obj.difference(
                                                    new_obj
                                                ).area
                                                if use_percent:
                                                    max_diff_pixels = (
                                                            best_obj.area
                                                            * max_diff_area
                                                            / 100
                                                    )
                                            if diff_area <= max_diff_pixels:
                                                # which one has the highest confidence
                                                g.logger.debug(
                                                    "high confidence match: '{}' is  in the same spot as the current best "
                                                    "detection '{}' based on '{}%' -> BEST: {} --- CURRENT: {}".format(
                                                        _l[new_idx],
                                                        _l_best_in_same_model[idx],
                                                        round(max_diff_area),
                                                        poly_bbox,
                                                        npoly_bbox,
                                                    )
                                                )
                                                # g.logger.debug(f"DBG=Y: {new_idx=} -- {_e=} ")
                                                if (
                                                        _c_best_in_same_model[idx]
                                                        >= _c[new_idx]
                                                ):
                                                    if (
                                                            _c_best_in_same_model[idx]
                                                            not in nb_c
                                                    ):
                                                        nb_b.append(
                                                            _b_best_in_same_model[idx]
                                                        )
                                                        nb_l.append(
                                                            _l_best_in_same_model[idx]
                                                        )
                                                        nb_c.append(
                                                            _c_best_in_same_model[idx]
                                                        )
                                                        nb_m.append(
                                                            _m_best_in_same_model[idx]
                                                        )
                                                        if (
                                                                not _e_best_in_same_model
                                                                == nb_e
                                                        ):
                                                            nb_e = copy.deepcopy(
                                                                _e_best_in_same_model
                                                            )
                                                        g.logger.debug(
                                                            f"high confidence match: for '"
                                                            f"{_l_best_in_same_model[idx]}' @ "
                                                            f"{_b_best_in_same_model[idx]} the current best model"
                                                            f" '{_m_best_in_same_model[idx]}' has the higher "
                                                            f"confidence -> {_c_best_in_same_model[idx]} > "
                                                            f"{_c[new_idx]} of '{_l[new_idx]}'"
                                                        )
                                                else:
                                                    if _c[new_idx] not in nb_c:
                                                        nb_b.append(_b[new_idx])
                                                        nb_l.append(_l[new_idx])
                                                        nb_c.append(_c[new_idx])
                                                        nb_m.append(_m[new_idx])
                                                        if (
                                                                not _e_best_in_same_model
                                                                == nb_e
                                                        ):
                                                            nb_e = copy.deepcopy(
                                                                _e_best_in_same_model
                                                            )
                                                        g.logger.debug(
                                                            f"high confidence match: for '"
                                                            f"{_l[new_idx]}' @ "
                                                            f"{_b[new_idx]} the current best model"
                                                            f" '{_m[new_idx]}' has the higher "
                                                            f"confidence -> {_c[new_idx]} than '{_m_best_in_same_model}'"
                                                            f" @ {_b_best_in_same_model} has "
                                                            f"{_c_best_in_same_model[idx]}"
                                                        )

                                            else:
                                                # high confidence match not needed here because they arent within the
                                                # configured difference apart
                                                if _c[new_idx] not in nb_c:
                                                    nb_b.append(_b[new_idx])
                                                    nb_l.append(_l[new_idx])
                                                    nb_c.append(_c[new_idx])
                                                    nb_m.append(_m[new_idx])
                                                    if not nb_e == _e:
                                                        nb_e = copy.deepcopy(_e)
                                                g.logger.debug(
                                                    4,
                                                    "high confidence match: '{}' has a difference in the area of best detection "
                                                    "of '{:.2f}', a minimum of {:.2f} is needed to not be considered "
                                                    "'in the same spot', high confidence filter disabled for this match".format(
                                                        _l[new_idx],
                                                        diff_area,
                                                        max_diff_pixels,
                                                    ),
                                                )
                                        elif best_obj.intersects(new_obj):
                                            g.logger.debug(
                                                f"high confidence match: best INTERSECTS current, COME AND "
                                                f"FIX ME 'MAN' plz=yes"
                                            )

                            if len(nb_l):
                                g.logger.debug(
                                    f"high confidence match: overriding BEST match in this sequence with "
                                    f"'{nb_m}' found '{nb_l}' ({nb_c}) @  {nb_b}"
                                )
                                _b_best_in_same_model = nb_b
                                _l_best_in_same_model = nb_l
                                _c_best_in_same_model = nb_c
                                _e_best_in_same_model = nb_e
                                _m_best_in_same_model = nb_m
# ---------------------------------------------------------------------
                        if (same_model_sequence_strategy == "first") and len(_b):
                            g.logger.debug(
                                3,
                                f"{lp} 'same_model_sequence_strategy'='first', Breaking out of sequence loop",
                            )
                            break
                # end of same model sequence iteration
                # at this state x_best_in_same_model contains the best match across
                # same model variations

                # still inside model loop
                if len(_l_best_in_same_model):
                    # g.logger.debug(f"{lp} BEST '{model_name}' matches for this frame ({self.media.last_frameid_read}"
                    #                f") so far={_l_best_in_same_model}->{_c_best_in_same_model}-->"
                    #                f"{_m_best_in_same_model}"
                    #                )
                    found = True
                    _labels_in_frame.extend(_l_best_in_same_model)
                    _boxes_in_frame.extend(_b_best_in_same_model)
                    _confs_in_frame.extend(_c_best_in_same_model)
                    _error_boxes_in_frame.extend(_e_best_in_same_model)
                    _detection_types_in_frame.extend(
                        [model_name] * len(_l_best_in_same_model)
                    )
                    _model_names_in_frame.extend(_m_best_in_same_model)
                    # polygons_in_frame.extend(_polygons_in_same_model)
                    # error_polygons_in_frame.extend(_error_polygons_in_same_model)
                    if frame_strategy == "first":
                        g.logger.debug(
                            2,
                            f"{lp} breaking out of MODEL loop as 'frame_strategy' is 'first'",
                        )
                        break
                else:
                    # g.logger.debug(f"{lp} no detections left after filtering")
                    if not filtered and self.model_valid:
                        g.logger.debug(
                            2,
                            f"{lp} no '{model_name}' matches at all in frame: {self.media.last_frame_id_read}",
                        )
                    elif filtered and self.model_valid:
                        g.logger.debug(
                            2,
                            f"{lp} all '{model_name}' matches in frame {self.media.last_frame_id_read} were filtered out",
                        )
                        filtered = False
                    else:
                        g.logger.debug(
                            f"only 1 detection and it wasn't filtered out? -- IDK MAN come and check it out"
                            f" {filtered = } {self.model_valid = }"
                        )

            # end of model loop

            # still in frame loop
            frame_timer_end = frame_perf_timer.stop_and_get_ms()
            g.logger.debug(
                2, f"perf:frame: {self.media.last_frame_id_read} took {frame_timer_end}"
            )
            if found:
                all_matches.append(
                    {
                        "labels": _labels_in_frame,
                        # "bbox2poly": polygons_in_frame,
                        "model_names": _model_names_in_frame,
                        "confidences": _confs_in_frame,
                        "detection_types": _detection_types_in_frame,
                        "frame_id": self.media.last_frame_id_read,
                        "boxes": _boxes_in_frame,
                        "error_boxes": _error_boxes_in_frame,
                        "image": frame.copy(),
                        # 'err2poly': error_polygons_in_frame,
                    }
                )
                all_frames.append(self.media.get_last_read_frame())
                # g.logger.debug(f"{lp} FRAME matches={_labels_in_frame} {_confs_in_frame} {_boxes_in_frame} {_model_names_in_frame}")
                if frame_strategy == "first":
                    g.logger.debug(
                        2,
                        f"{lp} breaking out of frame loop as 'frame_strategy' is 'first'",
                    )
                    break


        # end of while media loop
        # find best match in all_matches
        matched_poly, matched_err_poly = [], []

        for idx, item in enumerate(all_matches):
            # g.logger.debug(1,f"dbg:strategy: most:[{len(item['labels']) = } > {len(matched_l) = }] most_models:[{len(item['detection_types']) = } > {len(matched_detection_types) = }]"
            #                    f"most_unique:[{len(set(item['labels'])) = } > {len(set(matched_l)) = }]")
            if (
                    (frame_strategy == "first")
                    or (
                    (frame_strategy == "most")
                    and (len(item["labels"]) > len(matched_l))
                    )
                    or (
                    (frame_strategy == "most_models")
                    and (len(item["detection_types"]) > len(matched_detection_types))
                    )
                    or (
                    (frame_strategy == "most_unique")
                    and (len(set(item["labels"])) > len(set(matched_l)))
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
        if str2bool(mpd):
            # this is any mpd detections that were removed from this detections 'matched' lists, since we don't want to
            # be notified next time of the same thing we removed this detection, we will add it back to the mpd buffer
            if len(mpd_b):
                g.logger.debug(
                    4,
                    f"{lp}mpd: there are matched past detections that we need to add back into the buffer for"
                    f" the next detection -> {mpd_l} {mpd_b} {mpd_c}",
                )
                matched_b.extend(mpd_b)
                matched_l.extend(mpd_l)
                matched_c.extend(mpd_c)
            saved_event = repr(stream)
            pkl("write", matched_b, matched_l, matched_c, saved_event)
        mon_name = f"'Monitor': {g.config.get('mon_name')} ({g.mid})->'Event': "
        g.logger.debug(
            f"perf:{lp}FINAL: {mon_name if stream.isnumeric() else ''}"
            f"{stream} -> complete detection sequence took: {diff_time}"
        )
        self.media.stop()
        # if invoked again, we need to resize polys
        self.has_rescaled = False
        return matched_data, all_matches, all_frames
