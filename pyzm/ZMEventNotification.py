"""
ZMEventNotification
=====================
Implements a python implementation of the ZM ES server. 

"""
from typing import Optional

import websocket
import json
import time
import threading
import ssl
from pyzm.interface import GlobalConfig

g: Optional[GlobalConfig] = None


class ZMEventNotification:
    def __init__(self, options, globs):
        """Instantiates a thread that connects to the ZM Notification Server

        Args:
            globs (GlobalConfig): a pre instantiated Global configuration object that has a logger.
            options (dict): As below::
        
                {
                    'url': string # websocket url
                    'user': string # zm user name
                    'password': string # zm password
                    'allow_untrusted': boolean # set to true for self-signed certs
                    'on_es_message': callback function when a message is received
                    'on_es_close': callback function when the connection is closed
                    'on_es_error': callback function when an error occurs
                }
        
        Raises:
            ValueError: if no server is provided
        """

        if not options.get('url'):
            raise ValueError('ZMESClient: No server specified')

        self.url = options.get('url')
        self.user = options.get('user')
        self.password = options.get('password')
        self.allow_untrusted = options.get('allow_untrusted')

        self.on_es_message = options.get('on_es_message')
        self.on_es_close = options.get('on_es_close')
        self.on_es_error = options.get('on_es_error')

        self.ws = None
        self.queue = []

        self.connected = False
        self._disconnect = None
        self.worker_thread = None
        self.connect()

    def connect(self):
        """Connect to the ES
        """
        g.logger.info('ZMESClient: Connecting to ES')
        if not self.connected:
            self._disconnect = False
            self.worker_thread = threading.Thread(target=self._worker)
            self.worker_thread.start()
            g.logger.info('ZMESClient: Event Server init started')

    def disconnect(self):
        """Disconnect from the ES
        """
        g.logger.info('ZMESClient: Disconnecting from ES')
        if self.ws is not None:
            self._disconnect = True
            self.ws.keep_running = False
            if self.worker_thread is not None:
                self.worker_thread.join(15)
        # We have waited for the thread to finish
        # We are now disconnected
        self.connected = False
        self._disconnect = False

    def send(self, msg):
        """Send message to ES
        
        Args:
            msg (dict): message to send. The message should follow a control structure as specified in The
            `ES developer guide`_

        .. _ES developer guide:
                https://zmeventnotification.readthedocs.io/en/latest/guides/developers.html

        """
        if self.connected:
            self.ws.send(json.dumps(msg))
        else:
            g.logger.debug(1, f'ZMESClient: not yet connected, message queued[{len(self.queue)}]: {msg}')
            self.queue.append(msg)

    def _monkey_callback(self, callback, *args):
        """
        Monkey patch for WebSocketApp._callback() because it swallows
        exceptions.
        """
        if callback is not None:
            callback(self.ws, *args)

    def _worker(self):
        g.logger.info('ZMESClient: Inside Event Server thread, attempting to connect')
        ssl_opt = {}
        if self.allow_untrusted:
            ssl_opt['cert_reqs'] = ssl.CERT_NONE
            g.logger.warning('ZMESClient: Turning off certificate trust')

        while not self._disconnect:
            self.ws = websocket.WebSocketApp(self.url,
                                             on_message=lambda ws, msg: self._on_message(ws, msg),
                                             on_error=lambda ws, msg: self._on_error(ws, msg),
                                             on_close=lambda ws: self._on_close(ws),
                                             on_open=lambda ws: self._on_open(ws)
                                             )
            self.ws._callback = self._monkey_callback
            g.logger.info('ZMESClient: connected: ready to send/receive websocket messages')
            try:
                val = self.ws.run_forever(sslopt=ssl_opt)
                if not val:
                    break  # keyboard
            except Exception as e:
                g.logger.error(f'ZMESClient: Event Server Exception:{e}')
                # traceback.print_exc(file=sys.stdout)

            # The connection is aborted (intentional or not)
            self.connected = False
            self.ws.close()
            self.ws = None

            if not self._disconnect:
                g.logger.error('ZMESClient: run_forever() unexpectedly terminated')
                g.logger.info('ZMESClient: Will reconnect after 10 secs...')
                time.sleep(10)

        # Not connected anymore
        g.logger.info('ZMESClient: Exiting Event Server thread, correctly disconnected')

    def _on_open(self, ws):
        g.logger.info('ZMESClient: Sending auth info to ES')
        auth = {"event": "auth", "data": {"user": self.user, "password": self.password}}
        de_auth = {"event": "auth", "data": {"user": self.user, "password": '****'}}

        g.logger.debug(1, f'ZMESClient: Auth info to be sent: {de_auth}')
        ws.send(json.dumps(auth))

    def _on_message(self, ws, message):
        g.logger.info(f'ZMESClient: Got message from ES: {message}')
        message = json.loads(message)
        if message.get('event') == 'auth' and message.get('status') == 'Success':
            g.logger.info('ZMESClient: Auth accepted, connected state')
            self.connected = True
            while self.queue:
                msg = self.queue.pop(0)
                g.logger.debug(1, f'Sending pending message:{msg}')
                self.send(msg)

        if self.on_es_message:
            self.on_es_message(message)

    def _on_error(self, ws, error):
        g.logger.error(f'ZMESClient: Got error: {error}')
        if self.on_es_error:
            g.logger.info('invoking app error function and re-raising error')
            self.on_es_error(error)
            self.ws.close()
        raise error

    def _on_close(self, ws):
        g.logger.info('ZMESClient: Connection closed')
        if self.on_es_close:
            self.on_es_close()
