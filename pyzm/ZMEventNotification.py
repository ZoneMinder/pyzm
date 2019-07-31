"""
ZMEventNotification
=====================
Implements a python implementation of the ZM ES server. 

"""

import websocket
import json
import time
from pyzm.helpers.Base import Base
import threading
import ssl

class ZMEventNotification(Base):
    def __init__(self, options):
        """Instantiates a thread that connects to the ZM Notification Server

        Args:
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

        Base.__init__(self, options.get('logger'))
        if not options.get('url'):
            raise ValueError ('ZMESClient: No server specified')
        
        self.url = options.get('url')
        self.user = options.get('user')
        self.password = options.get('password')
        self.allow_untrusted = options.get('allow_untrusted')

        self.on_es_message = options.get('on_es_message')
        self.on_es_close = options.get('on_es_close')
        self.on_es_error = options.get('on_es_error')

        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()
        self.ready = False
        self.logger.Info ('ZMESClient: Event Server init started')
        self.queue = []
    
    def send(self, msg):
        """Send message to ES
        
        Args:
            msg (dict): message to send. The message should follow a control structure as specified in The `ES developer guide`_

        .. _ES developer guide:
                https://zmeventnotification.readthedocs.io/en/latest/guides/developers.html

        """
        if self.ready:
            self.ws.send(json.dumps(msg))
        else:
            self.logger.Debug (1,'ZMESClient: connection not yet ready, message queued[{}]: {}'.format(len(self.queue), msg))
            self.queue.append(msg)


    def _monkey_callback(self,callback, *args):
        """
        Monkey patch for WebSocketApp._callback() because it swallows
        exceptions.
        """
        if callback is not None:
            callback(self.ws, *args)
    
    def _worker(self):
        self.logger.Info('ZMESClient: Inside Event Server thread, attempting to connect')
        sslopt = {}
        if self.allow_untrusted:
            sslopt['cert_reqs'] = ssl.CERT_NONE
            self.logger.Warning('ZMESClient: Turning off certificate trust')
        self.ws = websocket.WebSocketApp(self.url, 
                                        on_message = lambda ws,msg:  self._on_message(ws, msg), 
                                        on_error = lambda ws,msg:  self._on_error(ws,msg), 
                                        on_close = lambda ws:  self._on_close(ws,msg),
                                        on_open = lambda ws: self._on_open(ws)
                                        )
        self.ws._callback  = self._monkey_callback
        while True:
            self.logger.Info ('ZMESClient: ready to send/receive websocket messages')
            try:
                val = self.ws.run_forever(sslopt=sslopt)
                if not val: break # keyboard
            except Exception as e:
                self.logger.Error ('ZMESClient: Event Server Exception:' + str(e))
                
                #traceback.print_exc(file=sys.stdout)


            self.logger.Error ('ZMESClient: run_forever() terminated' )
            self.logger.Info('ZMESClient: Will reconnect after 10 secs...')
            time.sleep(10)


    def _on_open(self, ws):   
        self.logger.Info('ZMESClient: Sending auth info to ES')
        auth={"event":"auth","data":{"user":self.user,"password":self.password}}
        self.logger.Debug(1, 'ZMESClient: Auth info to be sent: {}'.format(auth))
        ws.send(json.dumps(auth))

        

    def _on_message(self, ws, message):
        self.logger.Info('ZMESClient: Got message from ES: {}'.format(message))
        message = json.loads(message)
        if message.get('event') == 'auth' and message.get('status') == 'Success':
            self.logger.Info ('ZMESClient: Auth accepted, ready state')
            self.ready = True
            while self.queue:
                msg = self.queue.pop(0)
                self.logger.Debug (1, 'Sending pending message:{}'.format(msg))
                self.send(msg)

        if self.on_es_message: self.on_es_message(message)

    def _on_error(self, ws, error):
        self.logger.Error('ZMESClient: Got error: {}'.format(error))
        if self.on_es_error: 
            self.logger.Info('invoking app error function and re-raising error')
            self.on_es_error(error)
            self.ws.close()
        raise error
        
      

    def _on_close(self, ws):
       self.logger.Info ('ZMESClient: Connection closed')
       if self.on_es_close: self.on_es_close()

    



    
