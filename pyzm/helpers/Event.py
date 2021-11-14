"""
Event
======
Each Event will hold a single ZoneMinder Event.
It is basically a bunch of getters for each access to event data.
If you don't see a specific getter, just use the generic get function to get
the full object
"""
from typing import Optional, AnyStr
import requests
from pathlib import Path

from progressbar import ProgressBar as pb

g = None


class Event:
    def __init__(self, event=None, globs=None):
        global g
        g = globs
        self.event = event
        self.api = g.api

    @property
    def get(self) -> Optional[dict]:
        """There are 3 sections returned from an event request **'Event'**, **'Frame'** and **'Monitor'**.
           This will return the complete **'Event'** section which has information about this event.
            :returns: Dict containing all event information"""
        return self.event.get('Event')

    @property
    def get_Frames(self) -> Optional[list]:
        """There are 3 sections returned from an event request **'Event'**, **'Frame'** and **'Monitor'**.
           This will return the complete **'Frame'** section which has a list whose length corresponds to how many
           frames this event has along with information in each frame.
                    [
                    {'Id': '4971397', 'EventId': '29547', 'FrameId': '1', 'Type': 'Normal', 'TimeStamp': '2021-09-10 18:29:35', 'Delta': '0.00', 'Score': '0'},
                    {'Id': '4971501', 'EventId': '29547', 'FrameId': '105', 'Type': 'Normal', 'TimeStamp': '2021-09-10 18:29:42', 'Delta': '6.94', 'Score': '0'}
                    ]
            :returns: List of Dict containing all frames with their associated data for this event."""
        return self.event['Frame']

    @property
    def get_Monitor(self) -> Optional[dict]:
        """There are 3 sections returned from an event request **'Event'**, **'Frames'** and **'Monitor'**.
           This will return the complete 'Monitor' section which has information about the monitor
           that created this event.
            :returns: Dict containing monitor information"""
        return self.event['Monitor']

    @property
    def get_image_url(self, fid='snapshot'):
        """Returns the image URL for the specified frame
        
        Args:
            fid (str, optional): Frame ID to grab. Defaults to 'snapshot'.
        
        Returns:
            string: URL for the image
        """        
        eid = self.id()
        url = f"{self.api.portal_url}/index.php?view=image&eid={eid}&fid={fid}&{self.api.get_auth()}"
        return url

    @property
    def get_video_url(self) -> Optional[AnyStr]:
        """Returns the video URL for the specified event
        
        Returns:
            string: URL for the video file
        """        
        if not self.video_file():
            g.logger.error(f"Event '{self.id()}' does not have a video file, do you have 'Video Writer' set to "
                           f"'Encode' or 'Camera Passthrough'?"
                           )
            return None
        eid = self.id()
        url = self.api.portal_url+'/index.php?mode=mpeg&eid={}&view=view_video'.format(eid)+'&'+self.api.get_auth()
        return url
    
    def download_image(self, fid='snapshot', download_dir='.', show_progress=False):

        """Downloads an image frame of the current event object
        
        Args:
            fid (str, optional): Frame ID. Defaults to 'snapshot'.
            download_dir (str, optional): Path to save the image to. Defaults to '.'.
            show_progress (bool, optional): If enabled shows a progress bar (if possible). Defaults to False.
        
        Returns:
            string: path+filename of downloaded image
        
        """           
           
        url = self.get_image_url(fid)
        f = self._download_file(url, f"{self.id()}-{fid}.jpg", download_dir, show_progress)
        g.logger.info(f"Image file downloaded to '{f}'")
        return f

    def download_video(self, download_dir='.', show_progress=False):
        """Downloads a video mp4 of the current event object
        Only works if there is an actual video
        
        Args:
            download_dir (str, optional): Path to save the image to. Defaults to '.'.
            show_progress (bool, optional): If enabled shows a progress bar (if possible). Defaults to False.
        
        Returns:
            string: path+filename of downloaded video
        
        """           
           
        url = self.get_video_url
        if not url:
            return None
        f = self._download_file(url, f"{self.id()}-video.mp4", download_dir, show_progress)
        g.logger.info(f"Video file downloaded to {f}")
        return f

    def delete(self):
        """Deletes this event

        Returns:
            json: API response
        """
        url = f"{self.api.api_url}/events/{self.id()}.json"
        return self.api.make_request(url=url, type_action='delete')

    @property
    def monitor_id(self):
        """returns monitor ID of event object.

        Returns:
            int: monitor id
        """
        return int(self.event['Event']['MonitorId'])

    @property
    def start_time(self) -> AnyStr:
        """returns start time of event.

        Returns:
            str: start time
        """
        # print(f"{self.event = }")
        return self.event['Event']['StartTime']

    @property
    def id(self):
        """returns event id of event.

        Returns:
            int: event id
        """
        return int(self.event['Event']['Id'])

    @property
    def name(self):
        """returns name of event.

        Returns:
            string: name of event
        """      
        return self.event['Event']['Name'] or None

    @property
    def video_file(self):
        """returns name of video file in which the event was stored.

        Returns:
            string: name of video file
        """
        return self.event['Event'].get('DefaultVideo')

    @property
    def cause(self):
        """returns event cause.

        Returns:
            string: event cause
        """ 
        return self.event['Event']['Cause'] or None

    @property
    def notes(self):
        """returns event notes.

        Returns:
            string: event notes
        """
        return self.event['Event'].get('Notes')

    @property
    def fspath(self):
        """returns the filesystem path where the event is stored. Only
        available in ZM 1.33+

        Returns:
            string: path
        """
        return self.event['Event'].get('FileSystemPath')

    @property
    def duration(self):
        """Returns duration of event in seconds.

        Returns:
            float: duration
        """
        return float(self.event['Event']['Length'])

    @property
    def total_frames(self):
        """Returns total frames in event.

        Returns:
            int: total frames
        """
        return int(self.event['Event']['Frames'])

    @property
    def alarmed_frames(self):
        """Returns total alarmed frames in event.

        Returns:
            int: total alarmed frames
        """
        return int(self.event['Event']['AlarmFrames'])

    @property
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
            'average': float(self.event['Event']['AvgScore']),
            'max': float(self.event['Event']['MaxScore'])
        }
    
    def _download_file(self, url, file_name, dest_dir, show_progress=False, reauth=True):
        if not Path(dest_dir).exists():
            Path(dest_dir).mkdir()

        full_path_to_file = Path(dest_dir/file_name)

        if full_path_to_file.exists():
            return full_path_to_file.absolute()

        g.logger.info(f"Downloading '{file_name}' from '{url}'")

        try:
            req = requests  # fall back
            if self.api and self.api.get_session():
                req = self.api.get_session()
                # print ("OVERRIDING")
            r = req.get(url, allow_redirects=True, stream=True)

#            r = requests.get(url, allow_redirects=True, stream=True)
        except requests.exceptions.HTTPError as err:
            g.logger.debug(f"Got download access error: {err}")
            if err.response.status_code == 401 and reauth:
                g.logger.debug('Retrying login once')
                req._relogin()
                g.logger.debug('Retrying failed request again...')
                return self._download_file(url, file_name, dest_dir, show_progress, reauth=False)
            else:
                raise err
        except Exception as all_ex:
            g.logger.error("Could not establish connection. Download failed.")
            g.logger.debug(f"Exception -> {all_ex}")
            return None

        if r.headers.get('Content-Type') and "text/html" in r.headers.get('Content-Type'):
            if reauth:
                g.logger.error('It seems we were redirected to login page while trying to download, trying one more time')
                return self._download_file(url, file_name, dest_dir, show_progress, reauth=False)
            else:
                g.logger.error('Failed trying to reauthorize, not trying again')
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
        bar: Optional[pb] = None
        if show_progress:
            if file_size:
                bar = pb(maxval=num_bars).start()
            else:
                bar = pb().start()
        
        if r.status_code != requests.codes.ok:
            g.logger.error("Error occurred while downloading file")
            return None

        count = 0
        
        with open(full_path_to_file, 'wb') as fp:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fp.write(chunk)
                if show_progress and bar:
                    bar.update(count)
                    count += 1
            fp.close()

        return full_path_to_file
