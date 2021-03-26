"""
utils
======
Set of utility functions
"""

from configparser import ConfigParser
import cv2
import numpy as np
import re
import time
import pyzm.helpers.globals as g


class Timer:
    def __init__(self, start_timer=True):
        self.started = False
        if start_timer:
            self.start()

    def restart(self):
        self.start()

    def start(self):
        self.start = time.perf_counter()
        self.started = True
        self.final_inference_time = 0

    def stop(self):
        self.started = False
        self.final_inference_time = time.perf_counter() - self.start

    def get_ms(self):
        if self.final_inference_time:
            return '{:.2f} ms'.format(self.final_inference_time * 1000)
        else:
            return '{:.2f} ms'.format((time.perf_counter() - self.start) * 1000)

    def stop_and_get_ms(self):
        if self.started:
            self.stop()
        return self.get_ms()

def read_config(file):
    config_file = ConfigParser(interpolation=None,inline_comment_prefixes='#')
    config_file.read(file)
    return config_file

# wtf is this?
def get(key=None, section=None, conf=None):
    if conf.has_option(section, key):
        return conf.get(section, key)
    else:
        return None


def template_fill(input_str=None, config=None, secrets=None):
    class Formatter(dict):
        def __missing__(self, key):
            return "MISSING-{}".format(key)

    res = input_str
    if config:
        #res = input_str.format_map(Formatter(config)).format_map(Formatter(config))
        p = r'{{(\w+?)}}'
        res = re.sub(p, lambda m: config.get(m.group(1), 'MISSING-{}'.format(m.group(1))), res)
    if secrets:
        p = r'!(\w+)'
        res = re.sub(p, lambda m: secrets.get(m.group(1).lower(), '!{}'.format(m.group(1).lower())), res)
    return res

def draw_bbox(image=None,
              boxes=[],
              labels=[],
              confidences=[],
              polygons=[],
              box_color=None,
              poly_color=(255,255,255),
              poly_thickness = 1,
              write_conf=True):

        
        #print (1,"**************DRAW BBOX={} LAB={}".format(boxes,labels))
        slate_colors = [(39, 174, 96), (142, 68, 173), (0, 129, 254),
                        (254, 60, 113), (243, 134, 48), (91, 177, 47)]
        # if no color is specified, use my own slate
        if box_color is None:
            # opencv is BGR
            bgr_slate_colors = slate_colors[::-1]

        
        # first draw the polygons, if any
        newh, neww = image.shape[:2]
        image = image.copy()
        if poly_thickness:
            if not polygons:
                polygons=[]
            for ps in polygons:
                cv2.polylines(image, [np.asarray(ps['value'])],
                            True,
                            poly_color,
                            thickness=poly_thickness)

        # now draw object boundaries

        arr_len = len(bgr_slate_colors)
        for i, label in enumerate(labels):
            #=g.logger.Debug (1,'drawing box for: {}'.format(label))
            box_color = bgr_slate_colors[i % arr_len]
            if write_conf and confidences:
                label += ' ' + str(format(confidences[i] * 100, '.2f')) + '%'
            # draw bounding box around object

            #print ("DRAWING COLOR={} RECT={},{} {},{}".format(box_color, boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][3]))
            cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]),
                        box_color, 2)

            # write text
            font_scale = 0.8
            font_type = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 1
            #cv2.getTextSize(text, font, font_scale, thickness)
            text_size = cv2.getTextSize(label, font_type, font_scale,
                                        font_thickness)[0]
            text_width_padded = text_size[0] + 4
            text_height_padded = text_size[1] + 4

            r_top_left = (boxes[i][0], boxes[i][1] - text_height_padded)
            r_bottom_right = (boxes[i][0] + text_width_padded, boxes[i][1])
            cv2.rectangle(image, r_top_left, r_bottom_right, box_color, -1)
            #cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
            # location of text is botom left
            cv2.putText(image, label, (boxes[i][0] + 2, boxes[i][1] - 2), font_type,
                        font_scale, [255, 255, 255], font_thickness)

        return image
