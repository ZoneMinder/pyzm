# AWS Rekognition support for ZM object detection
# Author: Michael Ludvig

import boto3
import cv2
from typing import Optional
from pyzm.helpers.pyzm_utils import Timer
from pyzm.ml.object import Object

g: Optional[object] = None
lp: Optional[str] = None


class AwsRekognition(Object):
    def __init__(self, *args, **kwargs):
        global g, lp
        self.lp = lp = 'aws rek:'
        globs = kwargs['globs']
        g = globs
        self.options = kwargs['options']
        if self.options is None:
            raise ValueError(f'{lp} options must be provided')
        super().__init__(*args, **kwargs)

        self.sequence_name: str = ''
        self.lock_name: str = ''
        self.lock_timeout: int = 0
        self.is_locked: bool = False
        self.disable_locks: bool = False
        self.lock = None
        self.min_confidence = self.options.get('object_min_confidence', 0.7)
        if self.min_confidence < 1:  # Rekognition wants the confidence as 0% ~ 100%, not 0.00 ~ 1.00
            self.min_confidence *= 100

        # Extract AWS config values from options
        boto3_args = ('aws_region', 'aws_access_key_id', 'aws_secret_access_key')
        boto3_kwargs = {}
        for arg in boto3_args:
            if self.options.get(arg):
                boto3_kwargs[arg] = self.options.get(arg)
        if 'aws_region' in boto3_kwargs:
            boto3_kwargs['region_name'] = boto3_kwargs['aws_region']
            del boto3_kwargs['aws_region']

        self._rekognition = boto3.client('rekognition', **boto3_kwargs)
        g.logger.debug(2, f"{lp} AWS Rekognition initialised (min confidence: {self.min_confidence}%)")

    def detect(self, input_image=None):
        height, width = input_image.shape[:2]
        t = Timer()

        g.logger.debug(f"|---------- AWS Rekognition (image: {width}*{height}) ----------|")

        is_success, _buff = cv2.imencode(".jpg", input_image)
        if not is_success:
            g.logger.warning(f'{lp} unable to convert the image from numpy array / CV2 to JPG')
            return [], [], [], []
        image_jpg = _buff.tobytes()

        # Call AWS Rekognition
        response = self._rekognition.detect_labels(
            Image={'Bytes': image_jpg},
            MinConfidence=self.min_confidence
        )
        diff_time = t.stop_and_get_ms()
        g.logger.debug(2, f"perf:{lp} took {diff_time} -> detection response -> {response}")

        # Parse the returned labels
        bboxes = []
        labels = []
        confs = []

        for item in response['Labels']:
            if 'Instances' not in item:
                continue
            for instance in item['Instances']:
                if 'BoundingBox' not in instance or 'Confidence' not in instance:
                    continue
                label = item['Name'].lower()
                conf = instance['Confidence'] / 100
                bbox = (
                    round(width * instance['BoundingBox']['Left']),
                    round(height * instance['BoundingBox']['Top']),
                    round(width * (instance['BoundingBox']['Left'] + instance['BoundingBox']['Width'])),
                    round(height * (instance['BoundingBox']['Top'] + instance['BoundingBox']['Height']))
                )
                bboxes.append(bbox)
                labels.append(label)
                confs.append(conf)

        if labels:
            g.logger.debug(3, f"{lp} returning {labels} - {confs} - {bboxes}")
        else:
            g.logger.debug(f"{lp} no detections to return!")
        return bboxes, labels, confs, ['aws'] * len(labels)

    def acquire_lock(self):
        # AWS Rekognition doesn't need locking
        pass

    def release_lock(self):
        # AWS Rekognition doesn't need locking
        pass

    def get_model_name(self) -> str:
        return 'AWS Rek'

    def get_sequence_name(self) -> str:
        return self.sequence_name

    def get_options(self, key=None):
        if not key:
            return self.options
        else:
            return self.options.get(key)

    def load_model(self, *args, **kwargs):
        opts = kwargs['options']
        self.sequence_name = opts.get('name')
        g.logger.debug(f"{lp} Rekognition does not require a model to be loaded, setting sequence name to: "
                       f"{self.sequence_name}...")

    def get_classes(self):
        return
