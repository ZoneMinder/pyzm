"""
utils
======
Set of utility functions
"""

from pyzm.helpers.Base import ConsoleLog
from configparser import ConfigParser
import cv2
import numpy as np

def read_config(file):
    config_file = ConfigParser(interpolation=None)
    config_file.read(file)
    return config_file

def get(key=None, section=None, conf=None):
    if conf.has_option(section, key):
        return conf.get(section, key)
    else:
        return None

def draw_bbox(img,
              bbox,
              labels,
              confidence,
              polygons,
              box_color=None,
              poly_color=(255,255,255),
              poly_thickness = 1,
              write_conf=True):

        # g.logger.Debug (1,"DRAW BBOX={} LAB={}".format(bbox,labels))
        slate_colors = [(39, 174, 96), (142, 68, 173), (0, 129, 254),
                        (254, 60, 113), (243, 134, 48), (91, 177, 47)]
        # if no color is specified, use my own slate
        if box_color is None:
            # opencv is BGR
            bgr_slate_colors = slate_colors[::-1]

        
        # first draw the polygons, if any
        newh, neww = img.shape[:2]
        img = img.copy()
        if poly_thickness:
            for ps in polygons:
                cv2.polylines(img, [np.asarray(ps['value'])],
                            True,
                            poly_color,
                            thickness=poly_thickness)

        # now draw object boundaries

        arr_len = len(bgr_slate_colors)
        for i, label in enumerate(labels):
            #=g.logger.Debug (1,'drawing box for: {}'.format(label))
            box_color = bgr_slate_colors[i % arr_len]
            if write_conf and confidence:
                label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
            # draw bounding box around object

            #g.logger.Debug (1,"DRAWING RECT={},{} {},{}".format(bbox[i][0], bbox[i][1],bbox[i][2], bbox[i][3]))
            cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]),
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

            r_top_left = (bbox[i][0], bbox[i][1] - text_height_padded)
            r_bottom_right = (bbox[i][0] + text_width_padded, bbox[i][1])
            cv2.rectangle(img, r_top_left, r_bottom_right, box_color, -1)
            #cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
            # location of text is botom left
            cv2.putText(img, label, (bbox[i][0] + 2, bbox[i][1] - 2), font_type,
                        font_scale, [255, 255, 255], font_thickness)

        return img
