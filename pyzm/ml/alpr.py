from typing import Optional

import requests
import os
from glob import glob
import json
import subprocess
from pathlib import Path

import cv2
# Pycharm hack for intellisense
# from cv2 import cv2

from pyzm.helpers.pyzm_utils import id_generator, str2bool
from pyzm.interface import GlobalConfig
from pyzm.helpers.pyzm_utils import resize_image

g: Optional[GlobalConfig] = None
lp: str = '{lp}'


class AlprBase:
    def __init__(self, options=None, tempdir='/tmp', globs=None):
        global g
        g = globs
        if options is None:
            options = {}
        if not options.get('alpr_key') and options.get('alpr_service') != 'open_alpr_cmdline':
            g.logger.debug(2, '{lp} API key not specified and you are not using the command line ALPR, did you forget?')
        self.apikey = options.get('alpr_key')
        self.tempdir = tempdir
        self.url = options.get('alpr_url')
        self.options = options
        self.disable_locks = options.get('disable_locks', 'no')
        name = self.options.get('name') or 'ALPR'
        self.filename = None
        self.remove_temp = None
        # get rid of left over alpr temp files
        files = glob(f'{tempdir}/*-alpr.png')
        for file in files:
            os.remove(file)
        # g.logger.Debug(4, f"{lp} INITIALIZING '{name}' with options -> {self.options}")

    def get_options(self):
        return self.options

    def acquire_lock(self):
        pass

    def release_lock(self):
        pass

    def load_model(self):
        pass

    def set_key(self, key=None):
        self.apikey = key
        g.logger.debug(2, f"{lp} set_key-> key changed")

    def stats(self):
        g.logger.debug(f'{lp} stats not implemented in base class')

    def detect(self, input_image=None):
        g.logger.debug(f'{lp} detect not implemented in base class')

    def prepare(self, alpr_object):
        if not isinstance(alpr_object, str):
            g.logger.debug(
                f'{lp} the supplied object is not an absolute file path, assuming blob and creating file'
            )
            if self.options.get('max_size'):
                g.logger.debug(2, f"{lp} resizing image blob using max_size={self.options.get('max_size')}")
                vid_w = int(self.options.get('resize'))
                alpr_object = resize_image(alpr_object, vid_w)
            # use png so there is no loss
            self.filename = f"{self.tempdir}/{id_generator()}-alpr.png"
            cv2.imwrite(self.filename, alpr_object)
            self.remove_temp = True
        else:
            # If it is a file and zm_detect sent it, it would already be resized
            # If it is a file and zm_detect did not send it, no need to adjust scales
            # as there won't be a yolo/alpr size mismatch
            g.logger.debug(2, f"{lp} the supplied object is an absolute file path -> '{alpr_object}'")
            self.filename = alpr_object
            self.remove_temp = False

    def get_scale(self):
        if self.options.get('resize') and self.options.get('resize') != 'no':
            img = cv2.imread(self.filename)
            vid_w = int(self.options.get('resize'))
            old_h, old_w = img.shape[:2]
            img_new = resize_image(img, vid_w)
            new_h, new_w = img_new.shape[:2]
            rescale = True
            x_factor = new_w / old_w
            y_factor = new_h / old_h
            img = None
            img_new = None
            g.logger.debug(
                2,
                f'{lp} ALPR will use {old_w}x{old_h} but Object uses {new_w}x{new_h} so ALPR boxes'
                f' will be scaled {x_factor}x and {y_factor}y')
        else:
            x_factor = 1
            y_factor = 1
        return x_factor, y_factor


class Alpr(AlprBase):
    def __init__(self, options=None, tempdir='/tmp', globs=None):
        """Wrapper class for all ALPR objects

        Args:
            options (dict, optional): Config options. Defaults to {}.
            tempdir (str, optional): Path to store image for analysis. Defaults to '/tmp'.
        """
        global g
        g = globs
        AlprBase.__init__(self, options=options, tempdir=tempdir)
        if options is None:
            options = {}
        self.alpr_obj = None

        if self.options.get('alpr_service') == 'plate_recognizer':
            self.alpr_obj = PlateRecognizer(options=self.options)
        elif self.options.get('alpr_service') == 'open_alpr':
            self.alpr_obj = OpenAlpr(options=self.options)
        elif self.options.get('alpr_service') == 'open_alpr_cmdline':
            self.alpr_obj = OpenAlprCmdLine(options=self.options)
        else:
            raise ValueError(f"ALPR service '{self.options.get('alpr_service')}' not known")

    def detect(self, input_image=None):
        return self.alpr_obj.detect(input_image)

    def stats(self):
        return self.alpr_obj.stats()


class PlateRecognizer(AlprBase):
    # {lp}plate rec: API response JSON={'processing_time': 84.586, 'results': [{'box': {'xmin': 370, 'ymin': 171, 'xmax': 726, 'ymax': 310}, 'plate': 'cft4539'
    #   , 'region': {'code': 'ca-ab', 'score': 0.607}, 'score': 0.901, 'candidates': [{'score': 0.901, 'plate': 'cft4539'}], 'dscore': 0.76, 'vehicle': {'score': 0.244, 'type': 'Sedan', 'box': {'xmin': 49, 'ymin': 75,
    #    'xmax': 770, 'ymax': 418}}}], 'filename': '0517_HpKIJ_94bReShZAkFqUXRs-alpr.jpg', 'version': 1, 'camera_id': None, 'timestamp': '2021-09-12T05:17:16.788039Z'}]
    def __init__(self, options=None, tempdir='/tmp'):
        """Wrapper class for platerecognizer.com

        Args:
            options (dict, optional): Config options. Defaults to {}.
            tempdir (str, optional): Path to store image for analysis. Defaults to '/tmp'.
        """
        AlprBase.__init__(self, options=options, tempdir=tempdir)
        if options is None:
            options = {}
        url = self.options.get('alpr_url')
        if not url:
            self.url = 'https://api.platerecognizer.com/v1'
        g.logger.debug(
            f"{lp}plate rec: initialized with url: {self.url}")

    def stats(self):
        """Returns API statistics

        Returns:
            HTTP Response: HTTP response of statistics API
        """
        if self.options.get('alpr_api_type') != 'cloud':
            g.logger.debug(f'{lp}plate rec: local SDK does not provide stats')
            return {}
        try:
            if self.apikey:
                headers = {'Authorization': f"Token {self.apikey}"}
            else:
                headers = {}
            response = requests.get(
                f"{self.url}/statistics/",
                headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            response = {'error': str(e)}
        else:
            response = response.json()
        return response

    def detect(self, input_image=None):
        """Detects license plate using platerecognizer

        Args:
            input_image (image): image buffer

        Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """
        inf_object = input_image
        bbox = []
        labels = []
        confs = []
        self.prepare(inf_object)
        if str2bool(self.options.get('platerec_stats')):
            g.logger.debug(2, f'{lp}plate rec:  API usage stats: {json.dumps(self.stats())}')
        with open(self.filename, 'rb') as fp:
            try:
                platerec_url = self.url
                if self.options.get('alpr_api_type') == 'cloud':
                    platerec_url += '/plate-reader'

                platerec_payload = {}
                platerec_config = None
                if self.options.get('platerec_regions'):
                    platerec_payload['regions'] = self.options.get('platerec_regions')
                if self.options.get('platerec_payload'):
                    g.logger.debug(
                        '{lp}plate rec: found platerec_payload, overriding payload with values provided inside it')
                    platerec_payload = self.options.get('platerec_payload')
                if self.options.get('platerec_config'):
                    g.logger.debug('{lp}plate rec: found platerec_config, using it')
                    platerec_payload['config'] = json.dumps(self.options.get('platerec_config'))
                response = requests.post(
                    platerec_url,
                    timeout=15,
                    # self.url ,
                    files=dict(upload=fp),
                    data=platerec_payload,
                    headers={'Authorization': f"Token {self.apikey}"})
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                c = response.content
                response = {
                    'error':
                        f'Plate recognizer rejected the upload with: {e}.',
                    'results': []
                }
                g.logger.error(
                    f'{lp}plate rec: API rejected the upload with {e} and body:{c}'
                )
            else:
                response = response.json()
                g.logger.debug(3, f"{lp}plate rec: API response JSON={response}")

        # (xfactor, yfactor) = self.getscale()

        if self.remove_temp:
            os.remove(self.filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                dscore = plates['dscore']
                score = plates['score']
                if dscore >= self.options.get('platerec_min_dscore') and score >= self.options.get(
                        'platerec_min_score'):
                    x1 = round(int(plates['box']['xmin']))
                    y1 = round(int(plates['box']['ymin']))
                    x2 = round(int(plates['box']['xmax']))
                    y2 = round(int(plates['box']['ymax']))
                    labels.append(f'{lp}{label}')
                    bbox.append([x1, y1, x2, y2])
                    confs.append(plates['score'])
                else:
                    g.logger.debug(
                        f"{lp}plate rec: discarding plate:{label} because its dscore:{dscore}/score:{score} are not in "
                        f"range of configured dscore:{self.options.get('platerec_min_dscore')} score:"
                        f"{self.options.get('platerec_min_score')}")

        if len(labels):
            g.logger.debug(2, f"{lp}plate rec: Exiting ALPR with labels: {labels}")
        else:
            g.logger.debug(2, f'{lp}plate rec: Exiting ALPR with nothing detected')

        return bbox, labels, confs, ['platerec'] * len(labels)


class OpenAlpr(AlprBase):
    def __init__(self, options=None, tempdir='/tmp'):
        """Wrapper class for Open ALPR Cloud service

        Args:
            options (dict, optional): Various ALPR options. Defaults to {}.
            tempdir (str, optional): Temporary path to analyze image. Defaults to '/tmp'.
        """
        AlprBase.__init__(self, options=options, tempdir=tempdir)
        if options is None:
            options = {}
        if not self.url:
            self.url = "https://api.openalpr.com/v2/recognize"

        g.logger.debug(
            f"{lp} OpenALPR initialized with url: {self.url}")

    def detect(self, input_image=None):
        """Detection using OpenALPR

        Args:
            input_image (image): image buffer

        Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """
        alpr_object = input_image
        bbox = []
        labels = []
        confs = []

        self.prepare(alpr_object)
        with Path(self.filename).open('rb') as fp:
            try:
                params = ''
                if self.options.get('openalpr_country'):
                    params = f"{params}&country={self.options.get('openalpr_country')}"
                if self.options.get('openalpr_state'):
                    params = f"{params}&state={self.options.get('openalpr_state')}"
                if self.options.get('openalpr_recognize_vehicle'):
                    params = f"{params}&recognize_vehicle={str(self.options.get('openalpr_recognize_vehicle'))}"

                rurl = f"{self.url}?secret_key={self.apikey}{params}"
                g.logger.debug(2, f'Trying OpenALPR with url:{rurl}')
                response = requests.post(rurl, files={'image': fp})
                fp.close()
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                response = {
                    'error':
                        f"Open ALPR rejected the upload with {e}",
                    'results': []
                }
                g.logger.debug(
                    f"Open ALPR rejected the upload with {e}"
                )
            else:
                response = response.json()
                g.logger.debug(2, f"OpenALPR JSON: {response}")

        # (xfactor, yfactor) = self.getscale()

        rescale = False

        if self.remove_temp:
            os.remove(self.filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                conf = float(plates['confidence']) / 100
                if conf < float(self.options.get('openalpr_min_confidence')):
                    g.logger.debug(
                        f"OpenALPR: discarding plate: {label} because detected confidence {conf} is less than "
                        f"configured min confidence: {self.options.get('openalpr_min_confidence')}"
                    )
                    continue

                if plates.get('vehicle'):  # won't exist if recognize_vehicle is off
                    veh = plates.get('vehicle')
                    for attribute in ['color', 'make', 'make_model', 'year']:
                        if veh[attribute]:
                            label = label + ',' + veh[attribute][0]['name']

                x1 = round(int(plates['coordinates'][0]['x']))
                y1 = round(int(plates['coordinates'][0]['y']))
                x2 = round(int(plates['coordinates'][2]['x']))
                y2 = round(int(plates['coordinates'][2]['y']))
                labels.append(f"{lp}{label}")
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return bbox, labels, confs, ['openalpr'] * len(labels)


class OpenAlprCmdLine(AlprBase):
    def __init__(self, options=None, tempdir='/tmp'):
        """Wrapper class for OpenALPR command line utility

        Args:
            cmd (string, optional): The cli command. Defaults to None.
            options (dict, optional): Various ALPR options. Defaults to {}.
            tempdir (str, optional): Temporary path to analyze image. Defaults to '/tmp'.
        """
        if options is None:
            options = {}
        AlprBase.__init__(self, options=options, tempdir=tempdir)
        cmd = self.options.get('openalpr_cmdline_binary')
        self.cmd = f"{cmd} {self.options.get('openalpr_cmdline_params')}"
        if self.cmd.lower().find('-j') == -1:
            g.logger.debug(2, "{lp}cmdline: Adding -j to OpenALPR for JSON output")
            self.cmd = f"{self.cmd} -j"

    def detect(self, input_image=None):
        """Detection using OpenALPR command line

        Args:
            input_image (image): image buffer

         Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """
        i_object = input_image
        bbox = []
        labels = []
        confs = []
        from pyzm.helpers.pyzm_utils import Timer
        alpr_cmdline_exc_start = Timer()
        self.prepare(i_object)
        do_cmd = f"{self.cmd} {self.filename}"

        g.logger.debug(2, f"{lp}cmdline: executing: '{do_cmd}'")
        response = subprocess.check_output(do_cmd, shell=True)
        # response = subprocess.check_output(do_cmd)
        # this will cause the json.loads to fail if using gpu (haven't tested openCL)
        from re import sub
        p = b"--\(\!\)Loaded CUDA classifier\n"
        response = sub(p, b'', response)
        diff_time = alpr_cmdline_exc_start.stop_and_get_ms()
        g.logger.debug(2, f"perf:{lp}cmdline: took {diff_time}")
        g.logger.debug(2, f"{lp}cmdline: JSON response -> {response.decode('utf8')}")
        try:
            response = json.loads(response)
        except ValueError as e:
            g.logger.error(f"{lp}cmdline: Error parsing JSON response -> {e}")
            response = {}

        # (xfactor, yfactor) = self.getscale()

        rescale = False

        if self.remove_temp:  # move to BytesIO buffer?
            os.remove(self.filename)
        results = response.get('results')
        all_matches = response.get('candidates')
        # {"version":2,"data_type":"alpr_results","epoch_time":1631393388251,"img_width":800,"img_height":450,"processing_time_ms":501.42929
        #   1,"regions_of_interest":[{"x":0,"y":0,"width":800,"height":450}],
        #
        #   "results":[{"plate":"CFT4539","confidence":90.140419,"matches_template":0,"plate_index":0,"region":"","region_confidence":0,"processing_time_ms"
        #   :93.152191,"requested_topn":10,"coordinates":[{"x":412,"y":175},{"x":694,"y":180},{"x":694,"y":299},{"x":412,"y":295}],
        #
        #   "candidates":[{"plate":"CFT4539","confidence":90.140419,"matches_template":0},{"plate":"CF
        #   T4S39","confidence":82.398186,"matches_template":0},{"plate":"CFT439","confidence":79.333336,"matches_template":0},{"plate":"GFT4539","confidence":80.629532,"matches_template":0},{"plate":"CT4539","confidence"
        #   :80.943665,"matches_template":0},{"plate":"CPT4539","confidence":80.256454,"matches_template":0},{"plate":"CFT459","confidence":77.853737,"matches_template":0},{"plate":"CFT4B39","confidence":77.567482,"matche
        #   s_template":0},{"plate":"CF4539","confidence":75.923660,"matches_template":0}]}]}
        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                conf = float(plates['confidence']) / 100
                if conf < float(self.options.get('openalpr_cmdline_min_confidence')):
                    g.logger.debug(
                        f"{lp}cmdline: discarding plate: {label} ({conf}) is less than the configured min confidence "
                        f"-> '{self.options.get('openalpr_cmdline_min_confidence')}'")
                    continue
        # todo all_matches = 'candidates' and other data points from a successful detection via alpr_cmdline

                x1 = round(int(plates['coordinates'][0]['x']))
                y1 = round(int(plates['coordinates'][0]['y']))
                x2 = round(int(plates['coordinates'][2]['x']))
                y2 = round(int(plates['coordinates'][2]['y']))
                labels.append(f"{lp}{label}")
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return bbox, labels, confs, ['openalpr_cmd'] * len(labels)
