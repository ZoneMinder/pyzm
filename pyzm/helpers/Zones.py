"""
Zones
=======
Holds a list of Zone objects for a ZM configuration
"""
from pyzm.helpers.Zone import Zone
from typing import Optional, Union


allowed_raw = (
    'Zone[Name]',
    'Zone[MonitorId]',
    'Zone[Type]',
    'Zone[Units]',
    'Zone[NumCoords]',
    'Zone[Coords]',
    'Zone[Area]',
    'Zone[AlarmRGB]',
    'Zone[CheckMethod]',
    'Zone[MinPixelThreshold]',
    'Zone[MaxPixelThreshold]',
    'Zone[MinAlarmPixels]',
    'Zone[MaxAlarmPixels]',
    'Zone[FilterX]',
    'Zone[FilterY]',
    'Zone[MinFilterPixels]',
    'Zone[MaxFilterPixels]',
    'Zone[MinBlobPixels]',
    'Zone[MaxBlobPixels]',
    'Zone[MinBlobs]',
    'Zone[MaxBlobs]',
    'Zone[OverloadFrames]',
    'Zone[ExtendAlarmFrames]'
)


class Zones:
    def __init__(self, mid=None):
        global g
        g = GlobalConfig()

        self.zones = []
        self.api = g.api
        if mid:
            self.find_by_mon_id(int(mid))
        else:
            self._load()

    def _load(self, options: Optional[dict] = None):
        if options is None:
            options = {}
        g.logger.debug(2, 'Retrieving zones via API')
        url = f'{self.api.api_url}/zones/index.json'
        r = self.api.make_request(url=url, quiet=True)
        # g.logger.Debug(2,f"{r = }")
        zones = r.get('zones')
        self.zones = []
        for zone in zones:
            self.zones.append(Zone(zone=zone, api=self.api, globs=g))

    def list(self) -> dict:
        """Returns: a dict of dicts of zone information.

                **Return Format**:

                { 'zone name':
                        { 'id': '1', 'name':'zone name', etc. },
                'zone 2 name':
                        { 'id': '2', 'name':'zone 2 name', etc. }
                }
            """
        zones = {}
        for zone in self.zones:
            zones[zone.name().lower()] = zone.get()
        return zones

    def add(self, options: Optional[dict] = None) -> dict:
        """Creates a new zone.
            **Args:**
                    + *options* (dict)-> Set of attributes that define the Zone.
                    ``name`` ``type`` and ``monitorid`` are MANDATORY
                        - ``monitorid`` (int or str): monitor ID of zone.
                        - ``name`` (str): name of zone.
                        - ``type`` (str): 'type' of zone- > Active, Inactive, Preclusive, Exclusive, Inclusive, Privacy.
                        - ``unit`` (str): unit of measurement -> 'Pixels' or 'Percent'.
                        - ``coords`` (str): polygon coordinates -> 456,987 1076,700.
                        - ``area`` (str): number representing pixels or % of image this Zone occupies.
                        - ``min_alarm_pixels`` (str): number representing min pixels or % alarmed pixels.
                        - ``max_alarm_pixels`` (str): number representing max pixels or % alarmed pixels.
                        - ``min_filtered_pixels`` (str): number representing min pixels or % filtered pixels.
                        - ``max_filtered_pixels`` (str): number representing max pixels or % filtered pixels.
                        - ``filter_x`` (str): filtered pixels grouping box area -> '3', '5', '7', '9', '11', '13', '15'.
                        - ``filter_y`` (str): filtered pixels grouping box area -> '3', '5', '7', '9', '11', '13', '15'.
                        - ``min_blob_pixels`` (str): number representing min pixels or % area of blob matching pixels.
                        - ``max_blob_pixels`` (str): number representing max pixels or % area of blob matching pixels.
                        - ``max_blobs`` (str): number representing max number of blobs.
                        - ``min_blobs`` (str): number representing min number of blobs.
                        - ``overload_frames`` (str): number representing Alarm Overload Frames.
                        - ``extend_frames`` (str): number repreenting Extend Alarm Frames.
                        - ``raw`` (dict): Any RAW Zone value, Example: {
                                - ``Zone[AlarmRGB]``: '1278523',

                                - ``Zone[Name]``: 'My Driveway',
                                }
        :return: (dict): json response of API request
        """
        """
        https://server/zm/api/zones/add.json
        Zone[Name]=Jason-Newsted
        Zone[MonitorId]=3
        Zone[Type]=Active
        Zone[Units]=Percent
        Zone[NumCoords]=4
        Zone[Coords]=0,0 639,0 639,479 0,479
        Zone[Area]=307200
        Zone[AlarmRGB]=16711680
        Zone[CheckMethod]=Blobs
        Zone[MinPixelThreshold]=25
        Zone[MaxPixelThreshold]=
        Zone[MinAlarmPixels]=9216
        Zone[MaxAlarmPixels]=
        Zone[FilterX]=3
        Zone[FilterY]=3
        Zone[MinFilterPixels]=9216
        Zone[MaxFilterPixels]=230400
        Zone[MinBlobPixels]=6144
        Zone[MaxBlobPixels]=
        Zone[MinBlobs]=1
        Zone[MaxBlobs]=
        Zone[OverloadFrames]=0
        Zone[ExtendAlarmFrames]=0
        """
        url = f'{self.api.api_url}/zones/add.json'
        payload = {}

        if options.get('name'):
            payload['Zone[Name]'] = options.get('name')

        if options.get('monitorid'):
            mid_: str = options.get('monitorid')
            if mid_.isnumeric():
                payload['Zone[MonitorId]'] = mid_
            else:
                g.logger.error(f"Error: Zone 'monitorid' ({mid_}) only allowed numerical values")

        if options.get('type'):
            type_ = options.get('type')
            accepted = ('Active', 'Inactive', 'Preclusive', 'Exclusive', 'Inclusive', 'Privacy')
            if options.get('mode') in accepted:
                payload['Zone[Type]'] = options.get('mode')
            else:
                g.logger.error(f"Error: wrong Zone 'type'/'mode' ({type_}) only allowed -> {accepted}")

        if options.get('unit'):
            unit_ = options.get('unit')
            accepted = ('Percent', 'Pixels')
            if unit_ in accepted:
                payload['Zone[Unit]'] = unit_

        if options.get('coords'):
            coords_: str = options.get('coords')
            from pyzm_utils import str2tuple
            test = str2tuple(coords_)
            if test:
                payload['Zone[NumCoords]'] = str(len(coords_).split(' '))
                payload['Zone[Coords]'] = coords_

        if options.get('area'):
            area_: str = options.get('area')
            if area_.isnumeric():
                payload['Area'] = area_

        if options.get('zone_color'):
            alarmRGB: str = options.get('zone_color')
            if alarmRGB.isnumeric():
                payload['Zone[AlarmRGB]'] = alarmRGB

        if options.get('zone_method'):
            method_: str = options.get('zone_method')
            accepted = ['Blobs', 'AlarmedPixels', 'FilteredPixels']
            if method_ in accepted:
                payload['Zone[CheckMethod]'] = method_
            else:
                g.logger.error(f"Error: Zone 'CheckMethod' ({method_}), only allowed-> {accepted} ")

        if options.get('min_pixels'):
            minpixs: str = options.get('min_pixels')
            if minpixs.isnumeric():
                payload['Zone[MinPixelThreshold]'] = minpixs
        if options.get('max_pixels'):
            maxpixs = options.get('max_pixels')
            if maxpixs.isnumeric():
                payload['Zone[MaxPixelThreshold]'] = maxpixs

        if options.get('min_alarm_pixels'):
            min_alarm_pix = options.get('min_alarm_pixels')
            if min_alarm_pix.isnumeric():
                payload['Zone[MinAlarmPixels]'] = min_alarm_pix
        if options.get('max_alarm_pixels'):
            max_alarm_pix = options.get('max_alarm_pixels')
            if max_alarm_pix.isnumeric():
                payload['Zone[MaxAlarmPixels]'] = max_alarm_pix

        if options.get('max_filter_pixels'):
            max_fp = options.get('max_filter_pixels')
            if max_fp.isnumeric():
                payload['Zone[MaxFilterPixels]'] = max_fp
        if options.get('min_filter_pixels'):
            min_fp = options.get('min_filter_pixels')
            if min_fp.isnumeric():
                payload['Zone[MinFilterPixels]'] = min_fp
        if options.get('filter_x'):
            fx = options.get('filter_x')
            accepted = ['15', '13', '11', '9', '7', '5', '3']
            if fx in accepted:
                payload['Zone[FilterX]'] = fx
            else:
                g.logger.error(f"Error: Zone 'FilterX' ({fx}) not in allowed-> {accepted}")
        if options.get('filter_y'):
            fy = options.get('filter_y')
            accepted = ['15', '13', '11', '9', '7', '5', '3']
            if fy in accepted:
                payload['Zone[FilterY]'] = fy
            else:
                g.logger.error(f"Error: Zone 'FilterY' ({fy}) not in allowed-> {accepted}")

        if options.get('min_blob_pixels'):
            mbp = options.get('min_blob_pixels')
            if mbp.isnumeric():
                payload['Zone[MinBlobPixels]'] = mbp
        if options.get('max_blob_pixels'):
            mbp = options.get('max_blob_pixels')
            if mbp.isnumeric():
                payload['Zone[MaxBlobPixels]'] = mbp
        if options.get('max_blobs'):
            mb = options.get('max_blobs')
            if mb.isnumeric():
                payload['Zone[MaxBlobs]'] = mb
        if options.get('min_blobs'):
            mb = options.get('min_blobs')
            if mb.isnumeric():
                payload['Zone[MinBlobs]'] = mb

        if options.get('overload_frames'):
            o_frames = options.get('overload_frames')
            if o_frames.isnumeric():
                payload['Zone[OverloadFrames]'] = o_frames

        if options.get('extend_frames'):
            e_frames = options.get('overload_frames')
            if e_frames.isnumeric():
                payload['Zone[ExtendAlarmFrames]'] = e_frames

        if options.get('raw'):
            for k in options.get('raw'):
                if k in allowed_raw:
                    payload[k] = options.get('raw')[k]
                else:
                    g.logger.error(
                        f"Error: adding Zone using 'raw' options: only these option keys are allowed -> {allowed_raw}")

        if payload and len(payload):
            if payload.get('Zone[MonitorId]') and payload.get('Zone[Type]') and payload.get(
                    'Zone[Name]'):  # need at least these 3 things minimum

                return self.api.make_request(url=url, payload=payload, type_action='post')
            else:
                return {
                    'message': "Error: need a minimum of these 3 options to create a Zone: 'name' 'type' 'monitorid'"}

    def edit(self, id_=None, name: Optional[str] = None, options: Optional[dict] = None):
        """
        Edits an existing zone, after editing ZM must restart the process (zmc) for the Zones MonitorId, settings will take effect within 10-60 seconds due to a restart of the monitor the Zone is attached to.
            `Args:`
                - ``id`` (int or str): Id of Zone to edit.
                - ``name`` (str): Name of Zone to edit.
                - ``options`` (dict): Set of attributes to edit the Zone:
                            - ``monitorid`` (int or str): monitor ID of zone.
                            - ``name`` (str): name of zone.
                            - ``type`` (str): 'type' of zone- > Active, Inactive, Preclusive, Exclusive, Inclusive, Privacy.
                            - ``unit`` (str): unit of measurement -> 'Pixels' or 'Percent'.
                            - ``coords`` (str): polygon coordinates -> 456,987 1076,700.
                            - ``area`` (str): number representing pixels or % of image this Zone occupies.
                            - ``min_alarm_pixels`` (str): number representing min pixels or % alarmed pixels.
                            - ``max_alarm_pixels`` (str): number representing max pixels or % alarmed pixels.
                            - ``min_filtered_pixels`` (str): number representing min pixels or % filtered pixels.
                            - ``max_filtered_pixels`` (str): number representing max pixels or % filtered pixels.
                            - ``filter_x`` (str): filtered pixels grouping box area -> '3', '5', '7', '9', '11', '13', '15'.
                            - ``filter_y`` (str): filtered pixels grouping box area -> '3', '5', '7', '9', '11', '13', '15'.
                            - ``min_blob_pixels`` (str): number representing min pixels or % area of blob matching pixels.
                            - ``max_blob_pixels`` (str): number representing max pixels or % area of blob matching pixels.
                            - ``max_blobs`` (str): number representing max number of blobs.
                            - ``min_blobs`` (str): number representing min number of blobs.
                            - ``overload_frames`` (str): number representing Alarm Overload Frames.
                            - ``extend_frames`` (str): number repreenting Extend Alarm Frames.
                            - ``raw`` (dict): Any RAW Zone value, Example: {
                                    - ``Zone[AlarmRGB]``: '1278523',

                                    - ``Zone[Name]``: 'My Driveway',
                                    }

        :return: (dict): json response of API request
        """
        if id_ is None and name is None:
            g.logger.error(f"Error: no Zone ID or name passed")
            return False
        if id_ is None and (name and isinstance(name, str)):
            zone_: Zone = self.find(name=name)
            id_ = zone_.id()

        if options is None:
            options = {}
        url = f"{self.api.api_url}/zones/edit/{id_}.json"
        payload = {}
        if options.get('name'):
            payload['Zone[Name]'] = options.get('name')

        if options.get('monitorid'):
            mid_: str = options.get('monitorid')
            if mid_.isnumeric():
                payload['Zone[MonitorId]'] = mid_
            else:
                return g.logger.error(f"Error: Zone 'monitorid' ({mid_}) only allowed numerical values")

        if options.get('type'):
            type_: str = options.get('type')
            filter_xy_accepted = ('active', 'inactive', 'preclusive', 'exclusive', 'inclusive', 'privacy')
            if type_.lower() in filter_xy_accepted:
                payload['Zone[Type]'] = type_.capitalize()
            else:
                return g.logger.error(f"Error: wrong Zone 'type'/'mode' ({type_}) only allowed -> {filter_xy_accepted}")

        if options.get('unit'):
            unit_: str = options.get('unit')
            filter_xy_accepted = ('percent', 'pixels')
            if unit_.lower() in filter_xy_accepted:
                payload['Zone[Unit]'] = unit_.capitalize()
            else:
                g.logger.error(f"pyzm:zones:edit: zones 'Unit' ({unit_} only allowed-> {filter_xy_accepted}")
        if options.get('coords'):
            coords_: str = options.get('coords')
            from pyzm_utils import str2tuple
            test = str2tuple(coords_)
            if test:
                payload['Zone[NumCoords]'] = str(str(len(coords_)).split(' '))
                payload['Zone[Coords]'] = coords_
            else:
                g.logger.error(f"pyzm:zones:edit: zones 'coords' ({coords_} only allowed-> '234,567 123,456 xxx,yyy')")

        if options.get('area'):
            area_: str = options.get('area')
            if area_.isnumeric():
                payload['Area'] = area_
        # 'zone_color': number representing RGB represeinting color of outline of analyzed motio0n blob areas
        if options.get('zone_color'):
            alarm_rgb: str = options.get('zone_color')
            if alarm_rgb.isnumeric():
                payload['Zone[AlarmRGB]'] = alarm_rgb
        # 'zone_method': 'Blobs' , 'AlarmedPixels' , 'FilteredPixels'
        if options.get('zone_method'):
            method_: str = options.get('zone_method')
            filter_xy_accepted: tuple = ('blobs', 'alarmedpixels', 'filteredpixels')
            if method_.lower() in filter_xy_accepted:
                payload['Zone[CheckMethod]'] = method_.capitalize()
            else:
                g.logger.error(f"Error: Zone 'CheckMethod' ({method_}), only allowed-> {filter_xy_accepted} ")
        # 'min_pixels': number representing minimum pixel difference threshold
        # 'max_pixels': number representing maximum pixel difference threshold
        if options.get('min_pixels'):
            min_pixels: str = options.get('min_pixels')
            if min_pixels.isnumeric():
                payload['Zone[MinPixelThreshold]'] = min_pixels
        if options.get('max_pixels'):
            max_pixels = options.get('max_pixels')
            if max_pixels.isnumeric():
                payload['Zone[MaxPixelThreshold]'] = max_pixels

        if options.get('min_alarm_pixels'):
            min_alarm_pix: str = options.get('min_alarm_pixels')
            if min_alarm_pix.isnumeric():
                payload['Zone[MinAlarmPixels]'] = min_alarm_pix
        if options.get('max_alarm_pixels'):
            max_alarm_pix: str = options.get('max_alarm_pixels')
            if max_alarm_pix.isnumeric():
                payload['Zone[MaxAlarmPixels]'] = max_alarm_pix

        if options.get('max_filter_pixels'):
            max_fp: str = options.get('max_filter_pixels')
            if max_fp.isnumeric():
                payload['Zone[MaxFilterPixels]'] = max_fp
        if options.get('min_filter_pixels'):
            min_fp: str = options.get('min_filter_pixels')
            if min_fp.isnumeric():
                payload['Zone[MinFilterPixels]'] = min_fp

        filter_xy_accepted: tuple = ('15', '13', '11', '9', '7', '5', '3')
        if options.get('filter_x'):
            fx: str = options.get('filter_x')
            if fx in filter_xy_accepted:
                payload['Zone[FilterX]'] = fx
            else:
                g.logger.error(f"Error: Zone 'FilterX' ({fx}) not in allowed-> {filter_xy_accepted}")
        if options.get('filter_y'):
            fy: str = options.get('filter_y')
            if fy in filter_xy_accepted:
                payload['Zone[FilterY]'] = fy
            else:
                g.logger.error(f"Error: Zone 'FilterY' ({fy}) not in allowed-> {filter_xy_accepted}")

        if options.get('min_blob_pixels'):
            min_bp: str = options.get('min_blob_pixels')
            if min_bp.isnumeric():
                payload['Zone[MinBlobPixels]'] = min_bp
        if options.get('max_blob_pixels'):
            mbp: str = options.get('max_blob_pixels')
            if mbp.isnumeric():
                payload['Zone[MaxBlobPixels]'] = mbp

        if options.get('max_blobs'):
            mb: str = options.get('max_blobs')
            if mb.isnumeric():
                payload['Zone[MaxBlobs]'] = mb
        if options.get('min_blobs'):
            min_b: str = options.get('min_blobs')
            if min_b.isnumeric():
                payload['Zone[MinBlobs]'] = min_b
        if options.get('overload_frames'):
            o_frames: str = options.get('overload_frames')
            if o_frames.isnumeric():
                payload['Zone[OverloadFrames]'] = o_frames
        if options.get('extend_frames'):
            e_frames: str = options.get('overload_frames')
            if e_frames.isnumeric():
                payload['Zone[ExtendAlarmFrames]'] = e_frames

        if options.get('raw'):
            raw_opts: dict = options.get('raw')
            for k in raw_opts:
                payload[k] = raw_opts[k]

        if payload and len(payload):
            ret = self.api.make_request(url=url, payload=payload, type_action='post')
            g.logger.debug(f"pyzm:zones:edit: api returned-> '{ret.get('message')}'")
            if ret.get('message') != 'The zone has been saved.':
                g.logger.error(f"Error editing zone-> {ret.get('message')}")
            else:
                return ret

    def __iter__(self):
        if self.zones:
            for zone in self.zones:
                yield zone
        # raise StopIteration

    def find_by_mon_id(self, mid):
        """Erases self.zones and recreates it with only zones from `mid`
                Args:
                    mid (int): Monitor ID # search for `mid` zones
        """
        url: str = f"{self.api.api_url}/zones/forMonitor/{mid}.json"
        g.logger.debug(2, f"Retrieving monitor '{mid}' zones via API")
        r: dict = self.api.make_request(url=url, quiet=True)
        # print(f"find by mon: {r=}")
        zones = r.get('zones')
        self.zones = []
        for zone in zones:
            print(f"find by mon: {zone = }")
            self.zones.append(Zone(zone=zone, api=self.api))

    def find(self, id_ = None, name: Optional[str] = None) -> Optional[Zone]:
        """Given an id or name, returns matching zone object

        Args:
            id_ (int, optional): Id of zone. Defaults to None.
            name (string, optional): name of zone. Defaults to None.

        Returns:
            :class:`pyzm.helpers.Zone`: Matching zone object
        """
        if not id_ and not name:
            return None
        match = None
        if id_:
            key = 'Id'
        else:
            key = 'Name'

        for zone in self.zones:
            if id_ and zone.id() == id_:
                match = zone
                break
            if name and zone.name().lower() == name.lower():
                match = zone
                break
        return match
