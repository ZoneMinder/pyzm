"""
Configs
=======
Hold all ZM Config data. No need for a separate config class
"""
from typing import Optional
from pyzm.interface import GlobalConfig

g: GlobalConfig


class Configs:
    def __init__(self):
        global g
        g = GlobalConfig()
        self.configs = None
        self._load()

    def _load(self):
        g.logger.debug("Retrieving config via API")
        url = f"{g.api.api_url}/configs.json"
        try:
            r = g.api.make_request(url=url)
        except Exception as exc:
            g.logger.error(f"Error retrieving configuration from API -> {exc}")
        else:
            self.configs = r.get("configs")

    def list(self):
        """Returns list of configuration

        Returns:
            :class:`pyzm.helpers.Configs: list of configs
        """
        return self.configs

    def find(self, id_=None, name=None):
        """Given an id or name of config, returns its value

        Args:
            id_ (int, optional): ID of config. Defaults to None.
            name (string, optional): Name of config. Defaults to None.

        Returns:
            dict: config value::
            {
                'id': int,
                'name': string,
                'value': string
            }
        """
        if not id_ and not name:
            return None
        match = None
        if id_:
            key = "Id"
        else:
            key = "Name"

        for config in self.configs:
            if id_ and config["Config"]["Id"] == id_:
                match = config
                break
            if name and config["Config"]["Name"].lower() == name.lower():
                match = config
                break
        return {
            "id": int(match["Config"]["Id"]),
            "name": match["Config"]["Name"],
            "value": match["Config"]["Value"],
        }

    def set(self, name=None, val=None):
        """Given a config name, change its value

        Args:
            name (string, optional): Name of config. Defaults to None.
            val (string, optional): Value of config. Defaults to None.

        Returns:
            json: json of API to change
        """
        if not val:
            return
        if not name:
            return
        url = g.api.api_url + "/configs/edit/{}.json".format(name)
        data = {"Config[Value]": val}
        return g.api.make_request(url=url, payload=data, type_action="put")
