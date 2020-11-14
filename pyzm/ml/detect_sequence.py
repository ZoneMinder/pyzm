from pyzm.helpers.Base import Base
import pyzm.ml.object as  ObjectDetect
import pyzm.ml.face as FaceDetect
import re
import datetime
from pyzm.helpers.Media import MediaStream
import cv2


class DetectSequence(Base):
    def __init__(self, logger=None, options={}):
        super().__init__(logger)
        self.model_sequence = options.get('general', {}).get('model_sequence', 'object').split(',')    
        self.ml_options = options
        self.models = {}

    def load_models(self, sequences):
        if not sequences:
            sequences = self.model_sequence
        for seq in sequences:
            if seq == 'object':
                self.models[seq] = []
                for obj_seq in self.ml_options.get(seq):
                    self.logger.Debug (2,'Loading model  type:{} with options:{}'.format(seq,obj_seq ))
                    self.models[seq].append(ObjectDetect.Object(options=obj_seq))
            elif seq == 'face':
                self.models[seq] = []
                for face_seq in self.ml_options.get(seq):
                    self.models[seq].append(FaceDetect.Face(options=face_seq))
            else:
                self.logger.Error ('Invalid model: {}'.format(seq))
                raise ValueError ('Invalid model: {}'.format(seq))

    def detect_stream(self, stream, options={}):
        """Implements detection on a video stream

        Args:
            stream (string): location of media (file, url or event ID)
            api (object): instance of the API if the stream need to route via ZM
            options (dict, optional): Various options that control the detection process. Defaults to {}:
            - 'download': boolean # if True, will download video before analysis
            - 'start_frame' int # Which frame to start analysis. Default 1.
            - 'frame_skip': int # Number of frames to skip in video (example, 3 means process every 3rd frame)
            - 'max_frames' : int # Total number of frames to process before stopping
            - 'pattern': string # regexp for objects that will be matched. 'strategy' key below will be applied
                         to only objects that match this pattern
            - 'frame_set': string with exact frames to use (comma separated)
            - 'strategy': string # various conditions to stop matching as below
                'first' # stop at first match (Default)
                'most' # match the frame that has the highest number of detected objects
                'most_unique' # match the frame that has the highest number of unique detected objects
            - 'resize': int # width to resize image, default 800
            Returns:
                boxes (array): list of bounding boxes for matched frame
                labels (array): list of labels for matched frame
                confidences (array): list of confidences for matched frame
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

        

        stream_options = options
        match_strategy = options.get('strategy', 'first')
        match_pattern = re.compile(options.get('pattern', '.*'))
        same_model_sequence_strategy = self.ml_options.get('general',{}).get('same_model_sequence_strategy','first')
        all_matches = []
        matched_b = []
        matched_l = []
        matched_c = []
        matched_frame_id = None
        matched_frame_img = None
        manual_locking = False

        if len(self.model_sequence) > 1:
            manual_locking = False
            self.logger.Debug(1,'Using automatic locking as we are switching between models')
        else:
            manual_locking = True
            self.logger.Debug(1,'Using manual locking as we are only using one model')
            for seq in self.model_sequence:
                self.ml_options[seq]['auto_lock'] = False
            

        start = datetime.datetime.now()
        media = MediaStream(stream,'video', stream_options )
        while media.more():
            frame = media.read()
            if frame is None:
                self.logger.Debug(1,'Ran out of frames to read')
                break
            #fname = '/tmp/{}.jpg'.format(media.get_last_read_frame())
            #print (f'Writing to {fname}')
            #cv2.imwrite( fname ,frame)
            b = l = c = []
            
            for seq in self.model_sequence:
                self.logger.Debug(3,'============ Frame: {} Running {} model in sequence =================='.format(media.get_last_read_frame(),seq))
                _b=_l=_c=[]
                if not self.models.get(seq):
                    self.load_models([seq])
                    if manual_locking:
                        self.models[seq].acquire_lock()

                _b_best_in_model = _l_best_in_model = _c_best_in_model = []
                cnt = 1
                for m in self.models[seq]:
                    self.logger.Debug(3,'--------- Frame:{} Running variation: #{} -------------'.format(media.get_last_read_frame(),cnt))
                    cnt +=1
                    _b,_l,_c, = m.detect(image=frame)
                    match = list(filter(match_pattern.match, _l))
                    _tb = []
                    _tl = []
                    _tc = []
                    for idx, label in enumerate(_l):
                        if label not in match:
                            continue
                        _tb.append(_b[idx])
                        _tl.append(label)
                        _tc.append(_c[idx])
                    _b = _tb
                    _l = _tl
                    _c = _tc
                    self.logger.Debug(4,'This model iteration inside {} found: labels: {},conf:{}'.format(seq, _l, _c))
                   
                    if  ((same_model_sequence_strategy == 'first') 
                    or ((same_model_sequence_strategy == 'most') and (len(_l) > len(_l_best_in_model))) 
                    or ((same_model_sequence_strategy == 'most_unique') and (len(set(_l)) > len(set(_l_best_in_model))))):
                        _b_best_in_model = _b
                        _l_best_in_model = _l
                        _c_best_in_model = _c
                    
                    if (same_model_sequence_strategy=='first'):
                        self.logger.Debug(3, 'breaking out of same model loop, as matches found and strategy is "first"')
                        break

                if (len(_l_best_in_model)):
                    b.extend(_b_best_in_model)
                    l.extend(_l_best_in_model)
                    c.extend(_c_best_in_model)
            
        
                #print ('LABELS {} BOXES {}'.format(l,b))
                f_b = []
                f_l = []
                f_c = []

                match = list(filter(match_pattern.match, _l))
                for idx, label in enumerate(_l):
                    if label not in match:
                        continue
                    f_b.append(_b[idx])
                    f_l.append(label)
                    f_c.append(_c[idx])
                l = f_l
                b = f_b
                c = f_c

                if not len(l):
                    continue
                #print ('Frame:{}, labels:{}'.format(frames_processed, l))
                all_matches.append (
                    {
                        'frame_id': media.get_last_read_frame(),
                        'boxes': b,
                        'labels': l,
                        'confidences': c
                    }
                )


          
            matched_frame_img = frame
            
            if (match_strategy=='first'):
                break
            
        # release resources
        diff_time = (datetime.datetime.now() - start).microseconds / 1000

        for item in all_matches:
             if  ((match_strategy == 'first') 
                or ((match_strategy == 'most') and (len(item['labels'] > len(matched_l))) 
                or ((match_strategy == 'most_unique') and (len(set(item['labels'])) > len(set(matched_l)))))):
                matched_b =item['boxes']
                matched_c = item['confidences']
                matched_l  = item['labels']           
                matched_frame_id = item['frame_id']

       
        if manual_locking:
            if self.face_model:
                self.face_model.release_lock()
            if self.object_model:
                self.object_model.release_lock()
        self.logger.Debug(
            1,'detection (with image loads) took: {} milliseconds to process {}'.format(diff_time, stream))
        media.stop()
        return matched_b, matched_l, matched_c, matched_frame_id, matched_frame_img, all_matches


        