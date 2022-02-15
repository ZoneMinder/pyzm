"""
States
=======
Holds a list of States for a ZM configuration
Given states are fairly static, maintains a cache of states
which can be overridden 
"""
import requests

from pyzm.helpers.State import State
from typing import Optional
from pyzm.interface import GlobalConfig

g: GlobalConfig


class States:
    def __init__(self, *args, **kwargs):
        global g
        g = GlobalConfig()
        self.states: list = []
        self._load()

    def __iter__(self):
        if self.states:
            for state in self.states:
                yield state

    def _load(self):
        lp: str = "states:"
        g.logger.debug(2, "{lp} retrieving states via API")
        url: str = f"{g.api.api_url}/states.json"
        r: requests.Response = g.api.make_request(url=url)
        states: Optional[dict] = r.get("states")
        for state in states:
            self.states.append(State(state=state))

    def list(self):
        """Returns list of state objects

        Returns:
            list of `pyzm.helpers.State`: list of state objects
        """
        return self.states

    def find(self, id_: Optional[int] = None, name: Optional[str] = None):
        """Return a state object that matches either and id or a name.

        Args:
            id_ (int, optional): ID of state. Defaults to None.
            name (string, optional): name of state. Defaults to None.

        Returns:
            :class:`pyzm.helpers.State`: State object that matches
        """
        if not id_ and not name:
            return None
        match: Optional[str] = None
        if id_:
            key = "Id"
        else:
            key = "Name"

        for state in self.states:
            if id_ and state.id() == id_:
                match = state
                break
            if name and state.name().lower() == name.lower():
                match = state
                break
        return match
