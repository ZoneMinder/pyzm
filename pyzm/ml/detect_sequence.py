"""
DetectSequence
=====================
Primary entry point to invoke machine learning classes in pyzm
It is recommended you only use DetectSequence methods and not
lower level interfaces as they may change drastically.
"""

from pyzm.helpers.Base import Base
import pyzm.helpers.utils as utils
from pyzm.helpers.utils import Timer

import re
import datetime
from pyzm.helpers.Media import MediaStream
import cv2
import traceback
from shapely.geometry import Polygon
import copy
import pyzm.helpers.globals as g
import pickle
import os


class DetectSequence(Base):
    def __init__(self, options={}, global_config={}):
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

                    - global_config (dict): Used by zm_detect and mlapi to pass
                      additional config parameters that may not be present in ml_config
 
        """
    
        self.has_rescaled = False # only once in init
        self.set_ml_options(options,force_reload=True)
        self.global_config = global_config
        #g.logger.Debug(1,'WAKANDA FOREVER!!!!!!!!!!!!!!!')
        
    def get_ml_options(self):
        return self.ml_options

    def set_ml_options(self,options, force_reload=False):
        """ Use this to change ml options later. Note that models will not be reloaded 
            unless you add force_reload=True
        """
        self.model_sequence = options.get('general', {}).get('model_sequence', 'object').split(',')    
        self.ml_options = options
        self.stream_options = None
        self.media = None
        self.ml_overrides = {}
        if force_reload:
            g.logger.Debug (1, "Resetting models, will be loaded on next run")
            self.models = {}


    def _load_models(self, sequences):
        #print (f'***** {sequences}')
        if not sequences:
            sequences = self.model_sequence
        for seq in sequences:
            try:
                if seq == 'object':
                    import pyzm.ml.object as  ObjectDetect
                    self.models[seq] = []
                    for ndx,obj_seq in enumerate(self.ml_options.get(seq,{}).get('sequence')):
                        if obj_seq.get('enabled') == 'no':
                            g.logger.Debug(4, 'Skipping {} as it is disabled'.format(obj_seq.get('name') or 'index:{}'.format(ndx)))
                            continue
                        try:
                            obj_seq['disable_locks'] = self.ml_options.get('general',{}).get('disable_locks', 'no')
                            g.logger.Debug(4,'Loading sequence: {}'.format(obj_seq.get('name') or 'index:{}'.format(ndx)))
                            g.logger.Debug (2,'Initializing model  type:{} with options:{}'.format(seq,obj_seq ))
                            self.models[seq].append(ObjectDetect.Object(options=obj_seq))
                        except Exception as e:
                            g.logger.Error('Error loading same model variation for {}:{}'.format(seq,e))
                            g.logger.Debug(2,traceback.format_exc())

                elif seq == 'face':
                    import pyzm.ml.face as FaceDetect
                    self.models[seq] = []
                    for ndx,face_seq in enumerate(self.ml_options.get(seq,{}).get('sequence')):
                        if face_seq.get('enabled') == 'no':
                            g.logger.Debug(4, 'Skipping {} as it is disabled'.format(face_seq.get('name') or 'index:{}'.format(ndx)))
                            continue
                        try:
                            face_seq['disable_locks'] = self.ml_options.get('general',{}).get('disable_locks', 'no')
                            self.models[seq].append(FaceDetect.Face(options=face_seq))
                        except Exception as e:
                            g.logger.Error('Error loading same model variation for {}:{}'.format(seq,e))
                            g.logger.Debug(2,traceback.format_exc())

                elif seq == 'alpr':
                    import pyzm.ml.alpr as AlprDetect
                    self.models[seq] = []
                    for alpr_seq in self.ml_options.get(seq,{}).get('sequence'):
                        
                        try:
                            alpr_seq['disable_locks'] = self.ml_options.get('general',{}).get('disable_locks', 'no')
                            self.models[seq].append(AlprDetect.Alpr(options=alpr_seq))
                        except Exception as e:
                            g.logger.Error('Error loading same model variation for {}:{}'.format(seq,e))
                            g.logger.Debug(2,traceback.format_exc())                        

                else:
                    g.logger.Error ('Invalid model: {}'.format(seq))
                    raise ValueError ('Invalid model: {}'.format(seq))
            except Exception as e:
                g.logger.Error('Error loading same model variation for {}:{}'.format(seq,e))
                g.logger.Debug(2,traceback.format_exc())
                continue            
    
    def _rescale_polygons(self, polygons, xfactor, yfactor):
        newps = []
        for p in polygons:
            newp = []
            for x, y in p['value']:
                newx = int(x * xfactor)
                newy = int(y * yfactor)
                newp.append((newx, newy))
            newps.append({'name': p['name'], 'value': newp, 'pattern': p['pattern']})
        g.logger.Debug(3,'resized polygons x={}/y={}: {}'.format(
            xfactor, yfactor, newps))
        return newps

    
    # once all bounding boxes are detected, we check to see if any of them
    # intersect the polygons, if specified
    # it also makes sure only patterns specified in detect_pattern are drawn
    def _process_past_detections(self,m,bbox, label, conf, mid, model_name):

        try:
            FileNotFoundError
        except NameError:
            FileNotFoundError = IOError

        if not mid:
            g.logger.Debug(1,
                'Monitor ID not specified, cannot match past detections')
            return bbox, label, conf
        image_path = self.global_config.get('image_path') or m.get_options().get('image_path')
        mon_file = '{}/monitor-{}-data.pkl'.format(image_path,mid)
        g.logger.Debug(2,'trying to load ' + mon_file)
        saved_bs=[]
        saved_ls=[]
        saved_cs=[]
        try:
            fh = open(mon_file, "rb")
            saved_bs = pickle.load(fh)
            saved_ls = pickle.load(fh)
            saved_cs = pickle.load(fh)
            fh.close()
        except FileNotFoundError:
            g.logger.Debug(1,'No history data file found for monitor {}'.format(mid))
            return bbox, label, conf
        except EOFError:
            g.logger.Debug(1,'Empty file found for monitor {}'.format(mid))
            g.logger.Debug (1,'Going to remove {}'.format(mon_file))
            try:
                os.remove(mon_file)
            except Exception as e:
                g.logger.Error (f'Could not delete: {e}')
                pass
        except Exception as e:
            g.logger.Error(f'Error in processPastDetection: {e}')
            #g.logger.Error('Traceback:{}'.format(traceback.format_exc()))
            return bbox, label, conf

        # load past detection
        global_use_percent = False
        global_max_diff_area = 0
        global_mda = self.global_config.get('past_det_max_diff_area')
        if global_mda:
            _m = re.match('(\d+)(px|%)?$', global_mda,re.IGNORECASE)
            if _m:
                global_max_diff_area = int(_m.group(1))
                global_use_percent = True if _m.group(2) is None or _m.group(2) == '%' else False
            else:
                g.logger.Error('past_det_max_diff_area misformatted: {}'.format(mda))


             # it's very easy to forget to add 'px' when using pixels
            if global_use_percent and (global_max_diff_area < 0 or global_max_diff_area > 100):
                g.logger.Error(
                    'past_det_max_diff_area must be in the range 0-100 when using percentages. Setting to 5%, was {}'
                    .format(self.global_config.get(label_max_diff_area)))
                global_max_diff_area=5
       
        #g.logger.Debug (1,'loaded past: bbox={}, labels={}'.format(saved_bs, saved_ls));
        g.logger.Debug (4, 'process_past_detections: Global use_percent:{}, max_diff_area:{}'.format(global_use_percent,global_max_diff_area))
        new_label = []
        new_bbox = []
        new_conf = []

        for idx, b in enumerate(bbox):

            max_diff_area = global_max_diff_area
            use_percent = global_use_percent

            label_max_diff_area= '{}_past_det_max_diff_area'.format(label[idx])
            mda = m.get_options().get(label_max_diff_area) or self.global_config.get(label_max_diff_area)
            if mda:
                g.logger.Debug(4, 'Found {}={}'.format(label_max_diff_area, mda))
                _m = re.match('(\d+)(px|%)?$',mda,re.IGNORECASE)
                if _m:
                    max_diff_area = int(_m.group(1))
                    use_percent = True if _m.group(2) is None or _m.group(2) == '%' else False
                else:
                    g.logger.Error('{} misformatted: {}'.format(label_max_diff_area,mda))


                # it's very easy to forget to add 'px' when using pixels
                if use_percent and (max_diff_area < 0 or max_diff_area > 100):
                    g.logger.Error(
                        '{} must be in the range 0-100 when using percentages. Setting to 5%, was {}'
                        .format(label_max_diff_area, self.global_config.get(label_max_diff_area)))
                    max_diff_area = 5

            # iterate list of detections
            old_b = b
            it = iter(b)
            b = list(zip(it, it))

            b.insert(1, (b[1][0], b[0][1]))
            b.insert(3, (b[0][0], b[2][1]))
            #g.logger.Debug (1,"Past detection: {}@{}".format(saved_ls[idx],b))
            #g.logger.Debug (1,'BOBK={}'.format(b))
            obj = Polygon(b)
            foundMatch = False
            for saved_idx, saved_b in enumerate(saved_bs):
                # compare current detection element with saved list from file
                if saved_ls[saved_idx] != label[idx]: continue
                it = iter(saved_b)
                saved_b = list(zip(it, it))
                saved_b.insert(1, (saved_b[1][0], saved_b[0][1]))
                saved_b.insert(3, (saved_b[0][0], saved_b[2][1]))
                saved_obj = Polygon(saved_b)
                max_diff_pixels = max_diff_area

                if saved_obj.intersects(obj):
                    if obj.contains(saved_obj):
                        diff_area = obj.difference(saved_obj).area
                        if use_percent:
                            max_diff_pixels = obj.area * max_diff_area / 100
                    else:
                        diff_area = saved_obj.difference(obj).area
                        if use_percent:
                            max_diff_pixels = saved_obj.area * max_diff_area / 100

                    if diff_area <= max_diff_pixels:
                        g.logger.Debug(1,
                            'past detection {}@{} approximately matches {}@{} removing'
                            .format(saved_ls[saved_idx], saved_b, label[idx], b))
                        foundMatch = True
                        break
                    else:
                        g.logger.Debug(4,'Diff area of:{} > max_diff_pixels:{} for {}@{}, allowing it'
                        .format(diff_area, max_diff_pixels,label[idx], b))
            if not foundMatch:
                new_bbox.append(old_b)
                new_label.append(label[idx])
                new_conf.append(conf[idx])

        return new_bbox, new_label, new_conf


    def _filter_detections(self,m, seq, box,label,conf, polygons, h,w):
        
        # show_prefix
        stripped_labels = label[:]
        for idx, l in enumerate(stripped_labels):
            if l.startswith('('):
                items = l.split(') ')
                stripped_labels[idx] = items[1] if len(items) == 2 else items[0]

        # remember this needs to occur after a frame
        # is read, otherwise we don't have dimensions

        #print ("************ POLY={}".format(polygons))

        global_max_object_area = 0
        mds=m.get_options().get('max_detection_size') or self.global_config.get('max_detection_size')
        if mds:  
                g.logger.Debug(3,'Max object size found to be: {}'.format(mds))
                # Let's make sure its the right size
                _m = re.match('(\d*\.?\d*)(px|%)?$', mds,
                            re.IGNORECASE)
                if _m:
                    global_max_object_area = float(_m.group(1))
                    if _m.group(2) == '%':
                        global_max_object_area = float(_m.group(1))/100.0*(h * w)
                        g.logger.Debug (2,'Converted {}% to {}'.format(_m.group(1), global_max_object_area))
                else:
                    g.logger.Error('max_detection_size misformatted: {} - ignoring'.format(
                        mds))

        if not polygons:
            oldh =self.media.image_dimensions()['original'][0]
            oldw = self.media.image_dimensions()['original'][1]

            polygons.append({
                'name': 'full_image',
                'value': [(0, 0), (oldw, 0), (oldw, oldh), (0, oldh)],
                #'value': [(0, 0), (5, 0), (5, 5), (0, 5)],
                'pattern': None

            })
            g.logger.Debug(2,'No polygons, adding full image polygon: {}'.format(polygons[0]))

       #print ("************ RES={}, RESIZE={}".format(self.has_rescaled, self.stream_options))

        if (not self.has_rescaled) and (self.stream_options.get('resize') != 'no'):
            self.has_rescaled = True
            oldh =self.media.image_dimensions()['original'][0]
            oldw = self.media.image_dimensions()['original'][1]
            newh =self.media.image_dimensions()['resized'][0]
            neww = self.media.image_dimensions()['resized'][1]
            if (oldh != newh) or (oldw != neww):
                polygons[:] = self._rescale_polygons(polygons, neww / oldw, newh / oldh)
            

        doesIntersect = False
        new_label = [] 
        new_bbox =[]
        new_conf = []
        new_err = []
        for idx,b in enumerate(box):
            max_object_area = global_max_object_area
            label_max_object_area= '{}_max_detection_size'.format(label[idx])
            moa = m.get_options().get(label_max_object_area) or self.global_config.get(label_max_object_area)
            if moa:
                g.logger.Debug(4,'Found {}={}'.format(label_max_object_area,moa))
                # Let's make sure its the right size
                _m = re.match('(\d*\.?\d*)(px|%)?$', moa,
                            re.IGNORECASE)
                if _m:
                    max_object_area = float(_m.group(1))
                    if _m.group(2) == '%':
                        max_object_area = float(_m.group(1))/100.0*(h * w)
                        g.logger.Debug (2,'Converted {}% to {}'.format(_m.group(1), max_object_area))
                else:
                    g.logger.Error('{} misformatted: {} - ignoring'.format(label_max_object_area,
                        moa))


            old_b = b
            it = iter(b)
            b = list(zip(it, it))
            b.insert(1, (b[1][0], b[0][1]))
            b.insert(3, (b[0][0], b[2][1]))
            obj = Polygon(b)
            if obj.area > max_object_area and max_object_area:
                g.logger.Debug (1,'Ignoring {}@{} as its area of {} > {}'.format(label[idx], b, obj.area, max_object_area))
                continue

            for p in polygons:
                poly = Polygon(p['value'])
                g.logger.Debug(3,"intersection: comparing object:{},{} to polygon:{},{}".format(label[idx],obj,p['name'],poly))

                if obj.intersects(poly):
                    if  p['pattern']:
                        g.logger.Debug(3, '{} polygon/zone has its own pattern of {}, using that'.format(p['name'],p['pattern']))
                        r = re.compile(p['pattern'])
                        match = list(filter(r.match, stripped_labels))
                    else:
                        '''
                        g.logger.Debug (1,'**********************************')
                        g.logger.Debug (1,'{}'.format(self.ml_overrides))
                        g.logger.Debug (1,'**********************************')
                        '''
                        if self.ml_overrides.get(seq,{}).get('pattern'):
                            match_pattern = self.ml_overrides.get(seq,{}).get('pattern')
                            g.logger.Debug(2,'Match pattern overridden to {} in ml_overrides'.format(match_pattern))
                        else:
                            match_pattern = self.ml_options.get(seq,{}).get('general',{}).get('pattern', '.*')
                            g.logger.Debug(2,'Using global match pattern: {}'.format(match_pattern))

                        r = re.compile(match_pattern)

                        match = list(filter(r.match, stripped_labels))

                    #if label[idx].startswith('face:') or label[idx].startswith('alpr:') or label[idx] in match:
                    if stripped_labels[idx] in match:
                        g.logger.Debug(3,'{} intersects object:{}[{}]'.format(
                            p['name'], label[idx], b))
                        new_label.append(label[idx])
                        new_bbox.append(old_b)
                        new_conf.append(conf[idx])
                    else:
                        new_err.append(old_b)

                        g.logger.Debug(3,
                            '{} intersects object:{}[{}] but does NOT match your detect pattern filter'
                            .format(p['name'], label[idx], b))
                    doesIntersect = True
                    break

            # out of poly loop
            if not doesIntersect:
                new_err.append(old_b)
                g.logger.Info('object:{} at {} does not fall into any polygons, removing...'.
                    format(label[idx], obj))
        # out of primary bbox loop
        #print ("NEW ERR IS {}".format(new_err))
        return new_bbox, new_label, new_conf, new_err


    def detect_stream(self, stream, options={}, ml_overrides={}):
        """Implements detection on a video stream

        Args:
            stream (string): location of media (file, url or event ID)
            ml_overrides(string): Ignore it. You will almost never need it. zm_detect uses it for ugly foo
            options (dict, optional): Various options that control the detection process. Defaults to {}:
            
                - delay (int): Delay in seconds before starting media stream
                - download (boolean): if True, will download video before analysis. Defaults to False
                - download_dir (string): directory where downloads will be kept (only applies to videos). Default is /tmp
                - start_frame (int): Which frame to start analysis. Default 1.
                - frame_skip: (int): Number of frames to skip in video (example, 3 means process every 3rd frame)
                - max_frames (int): Total number of frames to process before stopping
                - pattern (string): regexp for objects that will be matched. 'frame_strategy' key below will be applied to only objects that match this pattern
                - frame_set (string): comma separated frames to read. Example 'alarm,21,31,41,snapshot'
                  Note that if you are specifying frame IDs and using ZM, remember that ZM has a frame buffer
                  Default is 20, I think. So you may want to start at frame 21.
                - contig_frames_before_error (int): How many contiguous frames should fail before we give up on reading this stream. Default 5
                - max_attempts (int): Only for ZM indirection. How many times to retry a failed frame get. Default 1
                - sleep_between_attempts (int): Only for ZM indirection. Time to wait before re-trying a failed frame
                - disable_ssl_cert_check (bool): If True (default) will allow self-signed certs to work
                - save_frames (boolean): If True, will save frames used in analysis. Default False
                - save_analyzed_frames (boolean): If True, will save analyzed frames (with boxes). Default False
                - save_frames_dir (string): Directory to save analyzed frames. Default /tmp
                - frame_strategy: (string): various conditions to stop matching as below
                    - 'most_models': Match the frame that has matched most models (does not include same model alternatives) (Default)
                    - 'first': Stop at first match 
                    - 'most': Match the frame that has the highest number of detected objects
                    - 'most_unique' Match the frame that has the highest number of unique detected objects
           
                - resize (int): Width to resize image, default 800
                - polygons(object): object # set of polygons that the detected image needs to intersect
                
        Returns:
           - object: representing matched frame, consists of:

            - box (array): list of bounding boxes for matched frame
            - label (array): list of labels for matched frame
            - confidence (array): list of confidences for matched frame
            - id (int): frame id of matched frame
            - img (cv2 image): image grab of matched frame

           - array of objects:

            - list of boxes,labels,confidences of all frames matched

        Note:

        The same frames are not retrieved depending on whether you set
        ``download`` to ``True`` or ``False``. When set to ``True``, we use
        OpenCV's frame reading logic and when ``False`` we use ZoneMinder's image.php function
        which uses time based approximation. Therefore, the retrieve different frame offsets, but I assume
        they should be reasonably close.
            
        """

        
        self.ml_overrides = ml_overrides
        self.stream_options = options
        frame_strategy = self.stream_options.get('frame_strategy', 'most_models' )
        all_matches = []
        matched_b = []
        matched_e = []
        matched_l = []
        matched_c = []
        matched_models = []
        matched_frame_id = None
        matched_images=[]
        
        matched_frame_img = None
        manual_locking = False


        if len(self.model_sequence) > 1:
            manual_locking = False
            g.logger.Debug(3,'Using automatic locking as we are switching between models')
        else:
            manual_locking = True
            g.logger.Debug(3,'Using manual locking as we are only using one model')
            for seq in self.model_sequence:
                self.ml_options[seq]['auto_lock'] = False        
        t = Timer()
        media = MediaStream(stream,'video', self.stream_options )
        self.media = media

        polygons = copy.copy(self.stream_options.get('polygons',[]))

        
        # Loops across all frames
        while self.media.more():
            frame = self.media.read()
            if frame is None:
                g.logger.Debug(1,'Ran out of frames to read')
                break
            #fname = '/tmp/{}.jpg'.format(self.media.get_last_read_frame())
            #print (f'Writing to {fname}')
            #cv2.imwrite( fname ,frame)
            g.logger.Debug (1, 'perf: Starting for frame:{}'.format(self.media.get_last_read_frame()))
            _labels_in_frame = []
            _boxes_in_frame = []
            _error_boxes_in_frame = []
            _confs_in_frame = []
            _models_in_frame = []

            # For each frame, loop across all models
            found = False
            for seq in self.model_sequence:
                if seq not in self.ml_overrides.get('model_sequence',seq):
                    g.logger.Debug (1, 'Skipping {} as it was overridden in ml_overrides'.format(seq))
                    continue
                g.logger.Debug(1,'============ Frame: {} Running {} model in sequence =================='.format(self.media.get_last_read_frame(),seq))
                pre_existing_labels = self.ml_options.get(seq,{}).get('general',{}).get('pre_existing_labels')
                if pre_existing_labels:
                    g.logger.Debug(2,'Making sure we have matched one of {} in {} before we proceed'.format(pre_existing_labels, _labels_in_frame))
                    if not any(x in _labels_in_frame for x in pre_existing_labels):
                        g.logger.Debug(1,'Did not find pre existing labels, not running model')
                        continue

                if not self.models.get(seq):
                    try:
                        self._load_models([seq])
                        if manual_locking:
                            for m in self.models[seq]:
                                m.acquire_lock()
                    except Exception as e:
                        g.logger.Error('Error loading model for {}:{}'.format(seq,e))
                        g.logger.Debug(2,traceback.format_exc())
                        continue

                same_model_sequence_strategy = self.ml_options.get(seq,{}).get('general',{}).get('same_model_sequence_strategy', 'first')
                g.logger.Debug (3,'{} has a same_model_sequence strategy of {}'.format(seq, same_model_sequence_strategy))
                
                # start of same model iteration
                _b_best_in_same_model = []
                _l_best_in_same_model = [] 
                _c_best_in_same_model = []
                _e_best_in_same_model = []

                cnt = 1
                # For each model, loop across different variations
                for m in self.models[seq]:
                    g.logger.Debug(3,'--------- Frame:{} Running variation: #{} -------------'.format(self.media.get_last_read_frame(),cnt))
                    cnt +=1
                    try:
                        _b,_l,_c = m.detect(image=frame)
                        g.logger.Debug(4,'This model iteration inside {} found: labels: {},conf:{}'.format(seq, _l, _c))
                    except Exception as e:
                        g.logger.Error ('Error running model: {}'.format(e))
                        g.logger.Debug(2,traceback.format_exc())
                        continue

                    # Now let's make sure the labels match our pattern
                    h,w,_ = frame.shape
                    _b,_l,_c, _e = self._filter_detections(m, seq,_b,_l,_c, polygons, h, w)
                    if _e:
                        _e_best_in_same_model.extend(_e)
                    if not len(_l):
                        continue

                    # Now let's take past detections into consideration 
                    # let's remove past detections first, if enabled 
                
                    #g.logger.Debug(4,'REMOVE ME: Self m options: {}'.format(m.get_options()))
                    mpd = m.get_options().get('match_past_detections') or self.global_config.get('match_past_detection')
                    if mpd == 'yes' and self.stream_options.get('mid'):
                        # point detections to post processed data set
                        g.logger.Info('Removing matches to past detections for monitor:{}'.format(self.stream_options.get('mid')))
                        #g.logger.Debug (1,'BEFORE PAST: {}, {}, {}'.format(_b,_l,_c))

                        _b,_l,_c = self._process_past_detections(m,_b, _l, _c, self.stream_options.get('mid'),seq)
                        #g.logger.Debug (1,'AFTER PAST: {}, {}, {}'.format(_b,_l,_c))
                        if not len (_l):
                            continue
                        
                        # save current objects for future comparisons
                        if seq!='face':
                            g.logger.Debug(1,
                                'Saving detections for monitor {} for future match'.format(
                                    self.stream_options.get('mid')))
                            try:
                                image_path = self.global_config.get('image_path') or m.get_options().get('image_path')
                                mon_file = '{}/monitor-{}-data.pkl'.format(image_path,self.stream_options.get('mid'))
                                f = open(mon_file, "wb")
                                pickle.dump(_b, f)
                                pickle.dump(_l, f)
                                pickle.dump(_c, f)
                                f.close()
                            except Exception as e:
                                g.logger.Error(f'Error writing to {mon_file}, past detections not recorded:{e}')


                    if  ((same_model_sequence_strategy == 'first') 
                    or ((same_model_sequence_strategy == 'most') and (len(_l) > len(_l_best_in_same_model))) 
                    or ((same_model_sequence_strategy == 'most_unique') and (len(set(_l)) > len(set(_l_best_in_same_model))))):
                        _b_best_in_same_model = _b
                        _l_best_in_same_model = _l
                        _c_best_in_same_model = _c
                        _e_best_in_same_model = _e
                    elif same_model_sequence_strategy == 'union':
                        _b_best_in_same_model.extend(_b)
                        _l_best_in_same_model.extend(_l)
                        _c_best_in_same_model.extend(_c)
                        _e_best_in_same_model.extend(_e)
                    if _l_best_in_same_model and self.stream_options.get('save_analyzed_frames') and self.media.get_debug_filename():
                            d = self.stream_options.get('save_frames_dir','/tmp')
                            f = '{}/{}-analyzed-{}.jpg'.format(d,self.media.get_debug_filename(), media.get_last_read_frame())
                            g.logger.Debug (4, 'Saving analyzed frame: {}'.format(f))
                            a = utils.draw_bbox(frame,_b_best_in_same_model,_l_best_in_same_model,_c_best_in_same_model,self.stream_options.get('polygons'))
                            for _b in _e_best_in_same_model:
                                cv2.rectangle(a, (_b[0], _b[1]), (_b[2], _b[3]),
                                    (0,0,255), 1)
                            cv2.imwrite(f,a)
                    if (same_model_sequence_strategy=='first') and len(_b):
                        g.logger.Debug(3, 'breaking out of same model loop, as matches found and strategy is "first"')
                        break
                # end of same model sequence iteration
                # at this state x_best_in_model contains the best match across 
                # same model variations
                if _l_best_in_same_model:
                    found = True
                    _labels_in_frame.extend(_l_best_in_same_model)
                    _boxes_in_frame.extend(_b_best_in_same_model)
                    _confs_in_frame.extend(_c_best_in_same_model)
                    _error_boxes_in_frame.extend(_e_best_in_same_model)
                    _models_in_frame.append(seq)
                    if (frame_strategy == 'first'):
                        g.logger.Debug (2, 'Breaking out of main model loop as strategy is first')
                        break
                else:
                    g.logger.Debug(2,'We did not find any {} matches in frame: {}'.format(seq,self.media.get_last_read_frame()))

            # end of primary model sequence
            if found:
                all_matches.append (
                    {
                        'frame_id': self.media.get_last_read_frame(),
                        'boxes': _boxes_in_frame,
                        'error_boxes': _error_boxes_in_frame,
                        'labels': _labels_in_frame,
                        'confidences': _confs_in_frame,
                        'models': _models_in_frame
                        
                    }
                )
                matched_images.append(frame.copy())
                if (frame_strategy == 'first'):
                    g.logger.Debug(2,'Frame strategy is first, breaking out of frame loop')
                    break
                
               
        # end of while media loop   
           
        #print ('*********** MATCH_STRATEGY {}'.format(model_match_strategy))
        for idx,item in enumerate(all_matches):
            if  ((frame_strategy == 'first') or 
            ((frame_strategy == 'most') and (len(item['labels']) > len(matched_l))) or
            ((frame_strategy == 'most_models') and (len(item['models']) > len(matched_models))) or
            ((frame_strategy == 'most_unique') and (len(set(item['labels'])) > len(set(matched_l))))):
                matched_b =item['boxes']
                matched_e = item['error_boxes']
                matched_c = item['confidences']
                matched_l  = item['labels']            
                matched_frame_id = item['frame_id']
                matched_models = item['models']
                matched_frame_img = matched_images[idx]
       
        if manual_locking:
            for seq in self.model_sequence:
                for m in self.models[seq]:
                    m.release_lock()

        diff_time = t.stop_and_get_ms()

        g.logger.Debug(
            1,'perf: TOTAL detection sequence (with image loads) took: {}  to process {}'.format(diff_time, stream))
        self.media.stop()

        matched_data = {
            'boxes': matched_b,
            'error_boxes': matched_e,
            'labels': matched_l,
            'confidences': matched_c,
            'frame_id': matched_frame_id,
            'image_dimensions': self.media.image_dimensions(),
            #'type': matched_type,
            'image': matched_frame_img,
            'polygons': polygons
        }
        # if invoked again, we need to resize polys
        self.has_rescaled = False
        return matched_data, all_matches


        