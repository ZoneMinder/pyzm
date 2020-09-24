"""
Event
======
Each Event will hold a single ZoneMinder Event.
It is basically a bunch of getters for each access to event data.
If you don't see a specific getter, just use the generic get function to get
the full object
"""

from pyzm.helpers.Base import Base
import progressbar as pb
import requests
import os
import sys

class Event(Base):
    def __init__(self, event=None, api=None, logger=None):
        super().__init__(logger)
        self.event = event
        self.logger = logger
        self.api = api
               
    def get(self):
        """Returns event object.

        Returns:
            :class:`pyzm.helpers.Event`: Event object
        """
        return self.event['Event']
    
    def get_image_url(self,fid='snapshot'):
        """Returns the image URL for the specified frame
        
        Args:
            fid (str, optional): Default frame identification. Defaults to 'snapshot'.
        
        Returns:
            string: URL for the image
        """        
        eid = self.id()
        url = self.api.portal_url+'/index.php?view=image&eid={}&fid={}'.format(eid,fid)+'&'+self.api.get_auth()
        return url

    def get_video_url(self):
        """Returms the video URL for the specified event
        
        Returns:
            string: URL for the video file
        """        
        if not self.video_file():
            self.logger.Error ('Event {} does not have a video file'.format(self.id()))
            return None
        eid = self.id()
        url = self.api.portal_url+'/index.php?mode=mpeg&eid={}&view=view_video'.format(eid)+'&'+self.api.get_auth()
        return url
    
    def download_image(self, fid='snapshot', dir='.', show_progress=False):     
        """Downloads an image frame of the current event object
        
        Args:
            fid (str, optional): Frame ID. Defaults to 'snapshot'.
            dir (str, optional): Path to save the image to. Defaults to '.'.
            show_progress (bool, optional): If enabled shows a progress bar (if possible). Defaults to False.
        """           
           
        url = self.get_image_url(fid)
        f =  self._download_file(url, str(self.id())+'-'+fid+'.jpg', dir, show_progress)
        self.logger.Info('File downloaded to {}'.format(f))

    def download_video(self, dir='.', show_progress=False):
        """Downloads a video mp4 of the current event object
        Only works if there an actual video 
        
        Args:
            dir (str, optional): Path to save the image to. Defaults to '.'.
            show_progress (bool, optional): If enabled shows a progress bar (if possible). Defaults to False.
        """           
           
        url = self.get_video_url()
        if not url:
            return None
        f =  self._download_file(url, str(self.id())+'-video'+'.mp4', dir, show_progress)
        self.logger.Info('File downloaded to {}'.format(f))

    def delete(self):
        """Deletes this event

        Returns:
            json: API response
        """
        url = self.api.api_url+'/events/{}.json'.format(self.id())
        return self.api._make_request(url=url, type='delete')

    def monitor_id(self):
        """returns monitor ID of event object.

        Returns:
            int: monitor id
        """
        return int(self.event['Event']['MonitorId'])
    
    def id(self):
        """returns event id of event.

        Returns:
            int: event id
        """
        return int(self.event['Event']['Id'])

    def name(self):
        """returns name of event.

        Returns:
            string: name of event
        """      
        return self.event['Event']['Name'] or None
    
    def video_file(self):
        """returns name of video file in which the event was stored.

        Returns:
            string: name of video file
        """
        return self.event['Event'].get('DefaultVideo')
    
    def cause(self):
        """returns event cause.

        Returns:
            string: event cause
        """ 
        return self.event['Event']['Cause'] or None
    
    def notes(self):
        """returns event notes.

        Returns:
            string: event notes
        """
        return self.event['Event']['Notes'] or None
    
    def fspath(self):
        """returns the filesystem path where the event is stored. Only
        available in ZM 1.33+

        Returns:
            string: path
        """
        return self.event['Event'].get('FileSystemPath')
    
    def duration(self):
        """Returns duration of event in seconds.

        Returns:
            float: duration
        """
        return float(self.event['Event']['Length'])
    
    def total_frames(self):
        """Returns total frames in event.

        Returns:
            int: total frames
        """
        return int(self.event['Event']['Frames'])
    
    def alarmed_frames(self):
        """Returns total alarmed frames in event.

        Returns:
            int: total alarmed frames
        """
        return int(self.event['Event']['AlarmFrames'])
    
    def score(self):
        """Returns total, average and max scores of event.

        Returns:
            dict: As below::

            {
                'total': float,
                'average': float,
                'max': float
            }
        """
        return {
            'total': float(self.event['Event']['TotScore']),
            'average':float(self.event['Event']['AvgScore']),
            'max':float(self.event['Event']['MaxScore'])
        }
    
    def _download_file(self,url, file_name, dest_dir, show_progress=False, reauth=True):

        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        full_path_to_file = dest_dir + os.path.sep + file_name

        if os.path.exists(dest_dir + os.path.sep + file_name):
            return full_path_to_file

        self.logger.Info("Downloading " + file_name + " from " + url)

        try:
            r = requests.get(url, allow_redirects=True, stream=True)
        except requests.exceptions.HTTPError as err:
            self.logger.Debug(1, 'Got download access error: {}'.format(err), 'error')
            if err.response.status_code == 401 and reauth:
                self.logger.Debug (1, 'Retrying login once')
                self._relogin()
                self.logger.Debug (1,'Retrying failed request again...')
                return self._download_file(url, file_name, dest_dir, show_progress, reauth=False)
            else:
                raise err
        except:
            #e = sys.exc_info()[0]
            #print(e)
            self.logger.Error("Could not establish connection. Download failed")
            return None

        if r.headers.get('Content-Type') and "text/html" in r.headers.get('Content-Type'):
            if reauth:
                self.logger.Error ('We got redirected to login while trying to download, trying one more time')
                return self._download_file(url, file_name, dest_dir, show_progress, reauth=False)
            else:
                self.logger.Error ('Failed doing reauth, not trying again')
                return None
        
        if r.headers.get('Content-Length'):
            file_size = int(r.headers['Content-Length'])
        else:
            file_size = 0
        
        chunk_size = 1024
        if file_size:
            num_bars = round(file_size / chunk_size)
        else:
            num_bars = 0

        if show_progress:
            if file_size:
                bar = pb.ProgressBar(maxval=num_bars).start()
            else:
                bar = pb.ProgressBar().start()
        
        if r.status_code != requests.codes.ok:
            self.logger.Error("Error occurred while downloading file")
            return None

        count = 0
        
        with open(full_path_to_file, 'wb') as file:
            for chunk in  r.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                if show_progress:
                    bar.update(count)
                    count +=1

        return full_path_to_file
