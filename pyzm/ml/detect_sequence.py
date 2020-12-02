from pyzm.helpers.Base import Base
import pyzm.ml.object as  ObjectDetect
import pyzm.ml.face as FaceDetect
import pyzm.ml.alpr as AlprDetect

import re
import datetime
from pyzm.helpers.Media import MediaStream
import cv2
import traceback
from shapely.geometry import Polygon

class DetectSequence(Base):
    def __init__(self, logger=None, options={}):

        if not logger:
            logger = options.get('logger')

        super().__init__(logger)
        #self.logger.Debug(1,'WAKANDA FOREVER!!!!!!!!!!!!!!!')
        self.model_sequence = options.get('general', {}).get('model_sequence', 'object').split(',')    
        self.ml_options = options
        self.stream_options = None
        self.models = {}
        self.media = None
        self.has_rescaled = False

    def load_models(self, sequences):
        if not sequences:
            sequences = self.model_sequence
        for seq in sequences:
            if seq == 'object':
                self.models[seq] = []
                for obj_seq in self.ml_options.get(seq,{}).get('sequence'):
                    self.logger.Debug (2,'Loading model  type:{} with options:{}'.format(seq,obj_seq ))
                    self.models[seq].append(ObjectDetect.Object(options=obj_seq, logger=self.logger))
            elif seq == 'face':
                self.models[seq] = []
                for face_seq in self.ml_options.get(seq,{}).get('sequence'):
                    self.models[seq].append(FaceDetect.Face(options=face_seq, logger=self.logger))
            elif seq == 'alpr':
                self.models[seq] = []
                for alpr_seq in self.ml_options.get(seq,{}).get('sequence'):
                    self.models[seq].append(AlprDetect.Alpr(options=alpr_seq, logger=self.logger))

            else:
                self.logger.Error ('Invalid model: {}'.format(seq))
                raise ValueError ('Invalid model: {}'.format(seq))
    
    def _rescale_polygons(self, polygons, xfactor, yfactor):
        newps = []
        for p in polygons:
            newp = []
            for x, y in p['value']:
                newx = int(x * xfactor)
                newy = int(y * yfactor)
                newp.append((newx, newy))
            newps.append({'name': p['name'], 'value': newp, 'pattern': p['pattern']})
        self.logger.Debug(3,'resized polygons x={}/y={}: {}'.format(
            xfactor, yfactor, newps))
        return newps

    def _filter_patterns(self,box,label,conf, global_match_pattern, polygons):
        
        # remember this needs to occur after a frame
        # is read, otherwise we don't have dimensions

        #print ("************ POLY={}".format(polygons))
        if not polygons:
            oldh =self.media.image_dimensions()['original'][0]
            oldw = self.media.image_dimensions()['original'][1]

            polygons.append({
                'name': 'full_image',
                'value': [(0, 0), (oldw, 0), (oldw, oldh), (0, oldh)],
                #'value': [(0, 0), (5, 0), (5, 5), (0, 5)],
                'pattern': global_match_pattern

            })
            self.logger.Debug(2,'No polygons, adding full image polygon: {}'.format(polygons[0]))

       #print ("************ RES={}, RESIZE={}".format(self.has_rescaled, self.stream_options))

        if (not self.has_rescaled) and (self.stream_options.get('resize') != 'no'):
            self.has_rescaled = True
            oldh =self.media.image_dimensions()['original'][0]
            oldw = self.media.image_dimensions()['original'][1]
            newh =self.media.image_dimensions()['resized'][0]
            neww = self.media.image_dimensions()['resized'][1]
            if (oldh != newh) or (oldw != neww):
                polygons[:] = self._rescale_polygons(polygons, neww / oldw, newh / oldh)
            

        global_r = re.compile(global_match_pattern)
        doesIntersect = False
        global_match = list(filter(global_r.match, label))
        new_label = [] 
        new_bbox =[]
        new_conf = []
        for idx,b in enumerate(box):
            old_b = b
            it = iter(b)
            b = list(zip(it, it))
            b.insert(1, (b[1][0], b[0][1]))
            b.insert(3, (b[0][0], b[2][1]))
            obj = Polygon(b)

            for p in polygons:
                poly = Polygon(p['value'])
                self.logger.Debug(3,"intersection: comparing object:{},{} to polygon:{}".format(label[idx],obj,poly))

                if obj.intersects(poly):
                    if  p['pattern'] != global_match_pattern:
                        self.logger.Debug(3, '{} polygon/zone has its own pattern of {}, using that'.format(p['name'],p['pattern']))
                        r = re.compile(p['pattern'])
                        match = list(filter(r.match, label))
                    else:
                        match = global_match
                        
                    if label[idx].startswith('face:') or label[idx].startswith('alpr:') or label[idx] in match:
                        self.logger.Debug(3,'{} intersects object:{}[{}]'.format(
                            p['name'], label[idx], b))
                        new_label.append(label[idx])
                        new_bbox.append(old_b)
                        new_conf.append(conf[idx])
                    else:
                        self.logger.Info(
                            'discarding "{}" as it does not match your filters'.
                            format(label[idx]))
                        self.logger.Debug(3,
                            '{} intersects object:{}[{}] but does NOT match your detect pattern filter'
                            .format(p['name'], label[idx], b))
                    doesIntersect = True
                    break
            # out of poly loop
            if not doesIntersect:
                self.logger.Info(
                    'object:{} at {} does not fall into any polygons, removing...'.
                    format(label[idx], obj))
        # out of primary bbox loop
        return new_bbox, new_label, new_conf


    def detect_stream(self, stream, options={}):
        """Implements detection on a video stream

        Args:
            stream (string): location of media (file, url or event ID)
            api (object): instance of the API if the stream need to route via ZM
            options (dict, optional): Various options that control the detection process. Defaults to {}:
            - 'download': boolean # if True, will download video before analysis
            - 'download_dir': directory where downloads will be kept (only applies to videos). Default is /tmp
            - 'start_frame' int # Which frame to start analysis. Default 1.
            - 'frame_skip': int # Number of frames to skip in video (example, 3 means process every 3rd frame)
            - 'max_frames' : int # Total number of frames to process before stopping
            - 'pattern': string # regexp for objects that will be matched. 'strategy' key below will be applied
                         to only objects that match this pattern
            - 'frame_set': string with exact frames to use (comma separated)
            - 'strategy': string # various conditions to stop matching as below
                'most_models': # Match the frame that has matched most models (does not include same model alternatives) (Default)
                'first' # stop at first match 
                'most' # match the frame that has the highest number of detected objects
                'most_unique' # match the frame that has the highest number of unique detected objects
            - 'resize': int # width to resize image, default 800
            - 'polygons': object # set of polygons that the detected image needs to intersect
            Returns:
                box (array): list of bounding boxes for matched frame
                label (array): list of labels for matched frame
                confidence (array): list of confidences for matched frame
                id (int): frame id of matched frame
                img (cv2 image): image grab of matched frame
                all_matches (array of objects): list of boxes,labels,confidences of all frames matched

            Note:
            The same frames are not retrieved depending on whether you set
            ``download`` to ``True`` or ``False``. When set to ``True``, we use
            OpenCV's frame reading logic and when ``False`` we use ZoneMinder's image.php function
            which uses time based approximation. Therefore, the retrieve different frame offsets, but I assume
            they should be reasonably close.
            
        """

        

        self.stream_options = options
        match_strategy = options.get('strategy', 'most_models')
        all_matches = []
        matched_b = []
        matched_l = []
        matched_c = []
        matched_models = []
        matched_frame_id = None
        matched_images=[]
        
        matched_frame_img = None
        manual_locking = False


        if len(self.model_sequence) > 1:
            manual_locking = False
            self.logger.Debug(3,'Using automatic locking as we are switching between models')
        else:
            manual_locking = True
            self.logger.Debug(3,'Using manual locking as we are only using one model')
            for seq in self.model_sequence:
                self.ml_options[seq]['auto_lock'] = False
        
        

        start = datetime.datetime.now()
        media = MediaStream(stream,'video', self.stream_options )
        self.media = media

        global_match_pattern = self.stream_options.get('pattern', '.*')
        polygons = self.stream_options.get('polygons',[])

        
        # Loops across all frames
        while self.media.more():
            frame = self.media.read()
            if frame is None:
                self.logger.Debug(1,'Ran out of frames to read')
                break
            #fname = '/tmp/{}.jpg'.format(self.media.get_last_read_frame())
            #print (f'Writing to {fname}')
            #cv2.imwrite( fname ,frame)
        
            _labels_in_frame = []
            _boxes_in_frame = []
            _confs_in_frame = []
            _models_in_frame = []

            # For each frame, loop across all models
            for seq in self.model_sequence:
                self.logger.Debug(1,'============ Frame: {} Running {} model in sequence =================='.format(self.media.get_last_read_frame(),seq))
                pre_existing_labels = self.ml_options.get(seq,{}).get('general',{}).get('pre_existing_labels')
                if pre_existing_labels:
                    self.logger.Debug(2,'Making sure we have matched one of {} in {} before we proceed'.format(pre_existing_labels, _labels_in_frame))
                    if not any(x in _labels_in_frame for x in pre_existing_labels):
                        self.logger.Debug(1,'Did not find pre existing labels: {} defined, not running model'.format(pre_existing_labels))
                        continue

                if not self.models.get(seq):
                    try:
                        self.load_models([seq])
                        if manual_locking:
                            self.models[seq].acquire_lock()
                    except Exception as e:
                        self.logger.Error('Error loading model for {}:{}'.format(seq,e))
                        self.logger.Debug(2,traceback.format_exc())
                        continue

                same_model_sequence_strategy = self.ml_options.get(seq,{}).get('general',{}).get('same_model_sequence_strategy', 'first')
                self.logger.Debug (3,'{} has a same_model_sequence strategy of {}'.format(seq, same_model_sequence_strategy))
                
                # start of same model iteration
                _b_best_in_same_model = []
                _l_best_in_same_model = [] 
                _c_best_in_same_model = []
                cnt = 1
                # For each model, loop across different variations
                for m in self.models[seq]:
                    self.logger.Debug(3,'--------- Frame:{} Running variation: #{} -------------'.format(self.media.get_last_read_frame(),cnt))
                    cnt +=1
                    try:
                        _b,_l,_c = m.detect(image=frame)
                    except Exception as e:
                        self.logger.Error ('Error running model: {}'.format(e))
                        self.logger.Debug(2,traceback.format_exc())

                        continue

                    self.logger.Debug(4,'This model iteration inside {} found: labels: {},conf:{}'.format(seq, _l, _c))
                   
                    if  ((same_model_sequence_strategy == 'first') 
                    or ((same_model_sequence_strategy == 'most') and (len(_l) > len(_l_best_in_same_model))) 
                    or ((same_model_sequence_strategy == 'most_unique') and (len(set(_l)) > len(set(_l_best_in_same_model))))):
                        _b_best_in_same_model = _b
                        _l_best_in_same_model = _l
                        _c_best_in_same_model = _c
                    if (same_model_sequence_strategy=='first'):
                        self.logger.Debug(3, 'breaking out of same model loop, as matches found and strategy is "first"')
                        break
                # end of same model sequence iteration
                # at this state x_best_in_model contains the best match across 
                # same model variations

                # Now let's make sure the labels match our pattern
                _b_best_in_same_model,_l_best_in_same_model,_c_best_in_same_model = self._filter_patterns(_b_best_in_same_model,_l_best_in_same_model,_c_best_in_same_model, global_match_pattern, polygons)
                if not len(_l_best_in_same_model):
                    continue
                _labels_in_frame.extend(_l_best_in_same_model)
                _boxes_in_frame.extend(_b_best_in_same_model)
                _confs_in_frame.extend(_c_best_in_same_model)
                _models_in_frame.append(seq)

            # end of primary model sequence
            all_matches.append (
                {
                    'frame_id': self.media.get_last_read_frame(),
                    'boxes': _boxes_in_frame,
                    'labels': _labels_in_frame,
                    'confidences': _confs_in_frame,
                    'models': _models_in_frame
                    
                }
            )
            matched_images.append(frame.copy())
            
            if (match_strategy=='first'):
                break
        # end of while media loop   
        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        #print ('*********** MATCH_STRATEGY {}'.format(match_strategy))
        for idx,item in enumerate(all_matches):
            if  ((match_strategy == 'first') or 
            ((match_strategy == 'most') and (len(item['labels']) > len(matched_l))) or
            ((match_strategy == 'most_models') and (len(item['models']) > len(matched_models))) or
            ((match_strategy == 'most_unique') and (len(set(item['labels'])) > len(set(matched_l))))):
                matched_b =item['boxes']
                matched_c = item['confidences']
                matched_l  = item['labels']            
                matched_frame_id = item['frame_id']
                matched_models = item['models']
                matched_frame_img = matched_images[idx]
       
        if manual_locking:
            if self.face_model:
                self.face_model.release_lock()
            if self.object_model:
                self.object_model.release_lock()
        self.logger.Debug(
            1,'detection (with image loads) took: {} milliseconds to process {}'.format(diff_time, stream))
        self.media.stop()

        matched_data = {
            'boxes': matched_b,
            'labels': matched_l,
            'confidences': matched_c,
            'frame_id': matched_frame_id,
            'image_dimensions': self.media.image_dimensions(),
            #'type': matched_type,
            'image': matched_frame_img
        }
        return matched_data, all_matches


        