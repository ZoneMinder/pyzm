import websocket
import json
import time
from pyzm.helpers.Base import Base
import threading
import ssl

class ZMES(Base):
    def __init__(self, options):
        """Instantiates a thread that connects to the ZM Notification Server
        
        TBD: Work in progress don't use yet....

        Args:
            options (dict): As below::
                {
                    'url': string # websocket url
                    'user': string # zm user name
                    'password': string # zm password
                    'allow_untrusted': boolean # set to true for self-signed certs

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

        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()
        self.logger.Info ('ZMESClient: Event Server init started')
 
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
        
        while True:
            self.logger.Info ('ZMESClient: ready to send/receive websocket messages')
            try:
                val = self.ws.run_forever()
                if not val: break
            except Exception as e:
                self.logger.Error ('ZMESClient: Event Server Exception:' + str(e))
                pass
            self.logger.Info('ZMESClient: Will reconnect after 10 secs...')
            time.sleep(10)


    def _on_open(self, ws):   
        self.logger.Info('ZMESClient: Sending auth info to ES')
        auth={"event":"auth","data":{"user":self.user,"password":self.password}}
        self.logger.Debug(1, 'ZMESClient: Auth info to be sent: {}'.format(auth))
        ws.send(json.dumps(auth))
        
       


    def _on_message(self, ws, message):
        self.logger.Info('ZMESClient: Got message from ES: {}'.format(message))

    def _on_error(self, ws, message):
        self.logger.Error('ZMESClient: Got error: {}'.format(message))
      

    def _on_close(self, ws):
       self.logger.Info ('ZMESClient: Connection closed')

    



    
