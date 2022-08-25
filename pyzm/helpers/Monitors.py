"""
Monitors
=========
Holds a list of Monitors for a ZM configuration
Given monitors are fairly static, maintains a cache of monitors
which can be overridden
"""

from pyzm.helpers.Monitor import Monitor
from typing import Optional
from pyzm.interface import GlobalConfig

g: GlobalConfig


class Monitors:
    def __init__(self):
        global g
        g = GlobalConfig()
        self.monitors = []
        self._load()

    def __len__(self):
        if self.monitors:
            return len(self.monitors)
        else:
            return 0

    def __str__(self) -> Optional[str]:
        if self.monitors:
            ret_val = []
            for mon in self.monitors:
                ret_val.append(str(mon))
            return str(ret_val)
        else:
            return None

    def _load(self, options=None):
        if options is None:
            options = {}
        g.logger.debug(2, "Retrieving monitors via API")
        url = f"{g.api.api_url}/monitors.json"
        r = g.api.make_request(url=url)
        ms = r.get("monitors")
        for m in ms:
            self.monitors.append(Monitor(monitor=m))

    def __iter__(self):
        if self.monitors:
            for mon in self.monitors:
                yield mon

    def list(self):
        return self.monitors

    def add(self, options=None):
        """Adds a new monitor

        Args:
            options (dict): Set of attributes that define the monitor::

                {
                    'function': string # function of monitor
                    'name': string # name of monitor
                    'enabled': boolean
                    'protocol': string
                    'host': string
                    'port': int
                    'path': string
                    'width': int
                    'height': int
                    'raw': {
                        # Any other monitor value that is not exposed above. Example:
                        'Monitor[Colours]': '4',
                        'Monitor[Method]': 'simple'
                    }

                }

        Returns:
            json: json response of API request
        """
        if options is None:
            options = {}
        url = f"{g.api.api_url}/monitors.json"
        payload = {}
        if options.get("function"):
            payload["Monitor[Function]"] = options.get("function")
        if options.get("name"):
            payload["Monitor[Name]"] = options.get("name")
        if options.get("enabled"):
            enabled = "1" if options.get("enabled") else "0"
            payload["Monitor[Enabled]"] = enabled
        if options.get("protocol"):
            payload["Monitor[Protocol]"] = options.get("protocol")
        if options.get("host"):
            payload["Monitor[Host]"] = options.get("host")
        if options.get("port"):
            payload["Monitor[Port]"] = str(options.get("port"))
        if options.get("path"):
            payload["Monitor[Path]"] = options.get("path")
        if options.get("width"):
            payload["Monitor[Width]"] = str(options.get("width"))
        if options.get("height"):
            payload["Monitor[Height]"] = str(options.get("height"))

        if options.get("raw"):
            for k in options.get("raw"):
                payload[k] = options.get("raw")[k]

        if payload:
            return g.api.make_request(url=url, payload=payload, type_action="post")

    def find(self, id_=None, name=None):
        """Given an id or name, returns matching monitor object

        Args:
            id_ (int, optional): MonitorId of monitor. Defaults to None.
            name (string, optional): Monitor name of monitor. Defaults to None.

        Returns:
            :class:`pyzm.helpers.Monitor`: Matching monitor object
        """
        if not id_ and not name:
            return None
        match = None
        if id_:
            key = "Id"
        else:
            key = "Name"

        for mon in self.monitors:
            if id_ and mon.id() == id_:
                match = mon
                break
            if name and mon.name().lower() == name.lower():
                match = mon
                break
        return match
