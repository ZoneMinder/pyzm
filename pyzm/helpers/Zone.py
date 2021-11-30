# from pyzm import api as zm_api
from traceback import format_exc
from typing import Optional

g = None


class Zone:
    def __init__(self, api=None, zone=None, globs=None):
        global g
        g = globs
        if zone:
            self.zone = zone
        # self.api: zm_api = g.api
        self.api = g.api
        self.zones: list = []
        # g.logger.Debug(1,f"{self.zone = }")

    def _load(self, options=None):
        if options is None:
            options = {}
        g.logger.debug(2, 'Retrieving zones via API - ZONE.py')
        url = self.api.api_url + '/zones.json'
        r = self.api.make_request(url=url)
        zones = r.get('zones')
        for zone in zones:
            self.zones.append(Zone(zone=zone, globs=g))

    def get(self) -> str:
        """Return all of the zone data"""
        return self.zone['Zone']

    def all(self):
        """Return all of the zone data"""
        return self.get()

    # might as well
    def units(self):
        return self.zone['Zone']['Units']

    def num_coords(self):
        return self.zone['Zone']['NumCoords']

    def area(self):
        return self.zone['Zone']['Area']

    def alarm_rgb(self):
        return self.zone['Zone']['AlarmRGB']

    def check_method(self):
        return self.zone['Zone']['CheckMethod']

    def min_pixel_thresh(self):
        return self.zone['Zone']['MinPixelThreshold']

    def max_pixel_thresh(self):
        return self.zone['Zone']['MaxPixelThreshold']

    def min_alarm_pixels(self):
        return self.zone['Zone']['MinAlarmPixels']

    def max_alarm_pixels(self):
        return self.zone['Zone']['MaxAlarmPixels']

    def filter_x(self):
        return self.zone['Zone']['FilterX']

    def filter_y(self):
        return self.zone['Zone']['FilterY']

    def min_filter_pixels(self):
        return self.zone['Zone']['MinFilterPixels']

    def max_filter_pixels(self):
        return self.zone['Zone']['MaxFilterPixels']

    def min_blob_pixels(self):
        return self.zone['Zone']['MinBlobPixels']

    def max_blob_pixels(self):
        return self.zone['Zone']['MaxBlobPixels']

    def min_blobs(self):
        return self.zone['Zone']['MinBlobs']

    def max_blobs(self):
        return self.zone['Zone']['MaxBlobs']

    def overload_frames(self):
        return self.zone['Zone']['OverloadFrames']

    def extend_alarm_frames(self):
        return self.zone['Zone']['ExtendAlarmFrames']

    # thank god thats over

    def id(self) -> int:
        """Returns:int Zones ID #"""
        return int(self.zone['Zone']['Id'])

    def name(self) -> str:
        """Returns Zones name"""
        return self.zone['Zone']['Name']

    def type(self) -> str:
        """Returns Zones 'type' or 'mode' (ex. 'Active' or 'Inactive')"""
        return self.zone['Zone']['Type']

    def polygons(self, get_tuple=False) -> str or tuple:
        """
        Returns polygon coords in space-delimted comma-split string or a tuple, ready to process by Polygons
            Args:
                - get_tuple (bool): returns  `tuple` if `True` otherwise `str`
        """
        return self.coords(get_tuple)

    def coords(self, get_tuple=False) -> str or tuple:
        """
        Returns polygon coords in space-delimted comma-split string or a tuple ready to process by Polygons module
            Args:
                - get_tuple (bool): returns a tuple of the Zone Co ordinates ready to be processed by Shape.Polygons if `True` otherwise returns a string
        """
        if get_tuple:
            from pyzm_utils import str2tuple
            return str2tuple(self.zone['Zone']['Coords'])
        return self.zone['Zone']['Coords']

    def monitorid(self) -> int:
        """Returns Zone Monitor ID #"""
        return int(self.zone['Zone']['MonitorId'])

    def disable(self) -> object:
        """Set Zone 'type' to 'Inactive'"""
        return self.inactive()

    def inactive(self) -> dict:
        """Set Zone 'type' to 'Inactive'"""
        url = self.api.api_url + '/zones/edit.json'
        payload = {
            'Zone[Id]': self.zone.id(),
            'Zone[Type]': 'Inactive'
        }
        try:
            ret = self.api.make_request(url=url, payload=payload, type_action='post')
        except Exception as ex:
            g.logger.error(f"pyzm:zone:err_msg-> {ex}")
            g.logger.debug(1, f"traceback-> {format_exc()}")
            return {'message': 'ERROR', 'data': ex}
        else:
            return ret

    def enable(self) -> object:
        """Set Zone 'type' to 'Active'"""
        return self.active()

    def active(self):
        """Set Zone type to 'Active'"""
        url = self.api.api_url + '/zones/edit/' + str(self.id()) + '.json'
        payload = {'Zone[Id]': self.id(), 'Zone[Type]': 'Active'}
        try:
            ret = self.api.make_request(url=url, payload=payload, type_action='post')
        except Exception as ex:
            g.logger.error(f"pyzm:zone:err_msg-> {ex}")
            g.logger.debug(1, f"traceback-> {format_exc()}")
            return {'message': 'ERROR', 'data': ex}
        else:
            return ret

    def privacy(self):
        """Set Zone 'type' to 'Privacy'"""
        url = self.api.api_url + '/zones/edit/' + str(self.id()) + '.json'

        payload = {'Zone[Id]': self.id(), 'Zone[Type]': 'Privacy'}
        try:
            ret = self.api.make_request(url=url, payload=payload, type_action='post')
        except Exception as ex:
            g.logger.error(f"pyzm:zone:err_msg-> {ex}")
            g.logger.debug(1, f"traceback-> {format_exc()}")
            return
        else:
            return ret

    def inclusive(self):
        """Set Zone 'type' to 'Inclusive'"""
        url = self.api.api_url + '/zones/edit.json'
        payload = {'Zone[Id]': self.zone.id(), 'Zone[Type]': 'Inclusive'}
        try:
            ret = self.api.make_request(url=url, payload=payload, type_action='post')
        except Exception as ex:
            g.logger.error(f"pyzm:zone:err_msg-> {ex}")
            g.logger.debug(1, f"traceback-> {format_exc()}")
            return
        else:
            return ret

    def preclusive(self):
        """Set Zone 'type' to 'Preclusive'"""
        url = self.api.api_url + '/zones/edit.json'
        payload = {'Zone[Id]': self.zone.id(), 'Zone[Type]': 'Preclusive'}
        try:
            ret = self.api.make_request(url=url, payload=payload, type_action='post')
        except Exception as ex:
            g.logger.error(f"pyzm:zone:err_msg-> {ex}")
            g.logger.debug(1, f"traceback-> {format_exc()}")
            return
        else:
            return ret

    def exclusive(self):
        """Set Zone 'type' to 'Exclusive'"""
        url = self.api.api_url + '/zones/edit.json'
        payload = {'Zone[Id]': self.zone.id(), 'Zone[Type]': 'Exclusive'}
        try:
            ret = self.api.make_request(url=url, payload=payload, type_action='post')
        except Exception as ex:
            g.logger.error(f"pyzm:zone:err_msg-> {ex}")
            g.logger.debug(1, f"traceback-> {format_exc()}")
            return
        else:
            return ret

    def set_zone_type(self, set_to: str = "Active") -> dict or str:
        """
        Set Zone 'type'
            Args:
                - set_to (str): set zone to 1 of -> 'Active' 'Inactive' 'Preclusive' 'Exclusive' 'Inclusive' 'Privacy'

                Default: 'Active'

                Input is converted to lower string and then capitalized, so if you type -> inACTivE it becomes -> inactive and finally -> Inactive

        """
        url = self.api.api_url + '/zones/edit/' + str(self.id()) + '.json'
        accepted = ['active', 'inactive', 'preclusive', 'exclusive', 'inclusive', 'privacy']
        if set_to.lower() in accepted:
            payload = {'Zone[Type]': set_to.capitalize()}
            try:
                ret = self.api.make_request(url=url, payload=payload, type_action='post')
            except Exception as ex:
                g.logger.error(f"pyzm:zone:err_msg-> {ex}")
                g.logger.debug(1, f"traceback-> {format_exc()}")
                return
            else:
                return ret
        else:
            return g.logger.error(f"Error: wrong Zone 'type' ({set_to}) only allowed-> {accepted}")

    def delete(self):
        """Deletes zone

        Returns:
            json: API response
        """
        url = self.api.api_url + '/zones/delete/{}.json'.format(self.id())
        ret = self.api.make_request(url=url, type_action='delete')
        if ret == 'The zone could not be deleted. Please, try again.':
            return g.logger.error(f"Error: deleting zone Id: '{self.id()}' Name: {self.name()}")
        return ret

    """
    self.zone = {'Zone': {'Id': '2', 'MonitorId': '2', 'Name': 'Front Yard', 'Type': 'Active', 'Units': 'Pixels', 'NumCoords': '9', 'Coords': '478,142 701,178 703,575 39,573 65,265 187,231 144,192 238,173 281,205', 'Area': '253186', 'AlarmRGB': '16711680', 'CheckMethod': 'Blobs', 'MinPixelThreshold': '25', 'MaxPixelThreshold': None, 'MinAlarmPixels': '4000', 'MaxAlarmPixels': '159507', 'FilterX': '3', 'FilterY': '3', 'MinFilterPixels': '3494', 'MaxFilterPixels': '177230', 'MinBlobPixels': '2988', 'MaxBlobPixels': '151912', 'MinBlobs': '1', 'MaxBlobs': None, 'OverloadFrames': '0', 'ExtendAlarmFrames': '0'}}
    
    WRITE/EDIT ZONE TEMPLATE
    
    Zone[Name]=Jason-Newsted
    &Zone[MonitorId]=3
    &Zone[Type]=Active
    &Zone[Units]=Percent
    &Zone[NumCoords]=4
    &Zone[Coords]=0,0 639,0 639,479 0,479
    &Zone[Area]=307200\
    &Zone[AlarmRGB]=16711680\
    &Zone[CheckMethod]=Blobs\
    &Zone[MinPixelThreshold]=25\
    &Zone[MaxPixelThreshold]=\
    &Zone[MinAlarmPixels]=9216\
    &Zone[MaxAlarmPixels]=\
    &Zone[FilterX]=3\
    &Zone[FilterY]=3\
    &Zone[MinFilterPixels]=9216\
    &Zone[MaxFilterPixels]=230400\
    &Zone[MinBlobPixels]=6144\
    &Zone[MaxBlobPixels]=\
    &Zone[MinBlobs]=1\
    &Zone[MaxBlobs]=\
    &Zone[OverloadFrames]=0"
    """
