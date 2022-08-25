# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ssl
import traceback
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt_client

from pyzm.helpers.pyzm_utils import str2bool, get_image, read_config, id_generator
from pyzm.interface import GlobalConfig
from typing import Optional

g: GlobalConfig

wasConnected = False
Connected = False  # global variable for the state of the connection


def on_log(client, userdata, level, buf):
    g.logger.debug(1, f"mqtt:paho_log: {buf}")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        g.logger.debug(1, f"mqtt:connect: connected to broker with flags-> {flags}")
        global Connected, wasConnected  # Use global variable
        Connected = True  # Signal connection
        wasConnected = True
    else:
        g.logger.debug(f"mqtt:connect: connection failed with result code-> {rc}")


def on_publish(client, userdata, mid):
    g.logger.debug(1, f"mqtt:on_publish: message_id: {mid = }")


class Mqtt:
    """Create an MQTT object to publish (subscribe coming for es control)
    config: (dict)
    config_file: path to a config file to read
    secrets: same as config but for secrets

    """

    # todo **kwargs instead of all of this
    def __init__(
        self,
        config=None,
        broker_address: str = None,
        port=None,
        user=None,
        password=None,
        config_file=None,
        secrets_filename=None,
        secrets=None,
    ):
        global g
        (
            self.image,
            self.path,
            self.conn_wait,
            self.client,
            self.tls_ca,
            self.connected,
            self.config,
            self.secrets,
            self.conn_time,
        ) = (None, None, None, None, None, None, None, None, None)
        self.ssl_cert = ssl.CERT_REQUIRED  # start with strict cert checking/verification of CN
        self.tls_self_signed = False
        g = GlobalConfig()
        # config and secrets
        if config:
            self.config = config
        else:
            if config_file:
                self.config = read_config(config_file)
        if secrets:
            self.secrets = secrets
        else:
            if secrets_filename:
                self.secrets = read_config(secrets_filename)

        if not user:
            self.user = self.config.get("mqtt_user")
        if not password:
            self.password = self.config.get("mqtt_pass")

        # todo: add ws/wss support
        if not broker_address:
            self.broker = self.config.get("mqtt_broker")
        else:
            self.broker = broker_address
        if not port:  # if a port isnt specified use protocol defaults for insecure/secure
            port = 1883
            if self.config.get("tls_ca"):
                port = 8883

                self.tls_ca = self.config.get("tls_ca")
                if self.config.get("mqtt_tls_allow_self_signed"):
                    self.tls_self_signed = True
                    self.ssl_cert = ssl.CERT_NONE

        if not g.config.get("mqtt_port"):
            self.port = port
        else:
            self.port = g.config.get("mqtt_port", port)

        self.tls_insecure = self.config.get("mqtt_tls_insecure") if self.config.get("mqtt_tls_insecure") else None
        self.mtls_cert = self.config.get("tls_cert") if self.config.get("tls_cert") else None
        self.mtls_key = self.config.get("tls_key") if self.config.get("tls_key") else None
        self.retain = str2bool(self.config.get("mqtt_retain"))
        self.qos = self.config.get("mqtt_qos", 0)
        self.client_id = "zmes-"

    def isConnected(self):
        return Connected

    def create_ml_image(self, image_path=None, cause=None, image=None, _type="byte"):
        """Prepares an image to be published, tested on jpg and gif so far. Give it an image or a (path and cause: [
        s] dog:98%), *** image will take precedence if all 3 sent path and cause; determines if it returns
        alarm/snapshot/objdetect.jpg or objdetect.gif. precedence is
        objdetect.gif->objdetect.jpg->snapshot.jpg/alarm.jpg it then wraps the image in a bytearray and stores it
        internally waiting to publish to home assistant mqtt camera topic
        """
        if image:
            self.image = image
        else:
            if image_path and cause:
                self.path = image_path
                self.path = get_image(self.path, cause)
                if _type == "byte":
                    g.logger.debug(
                        f"mqtt:grab_image: {Path(self.path).suffix}"
                        f" to be used is: '{self.path}', converting to byte array"
                    )
                    with open(self.path, "rb") as fd:
                        self.image = bytearray(fd.read())
                else:
                    import base64

                    g.logger.debug(
                        f"mqtt:grab_image: {Path(self.path).suffix}"
                        f" to be used is: '{self.path}', converting to BASE64"
                    )
                    with open(self.path, "rb") as fd:
                        self.image = base64.b64encode(fd.read()).decode("utf-8")

    def get_options(self):
        return {
            "client_id": self.client_id,
            "broker": self.broker,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "retain_published": self.retain,
            "tls_info": {
                "self_signed": self.tls_self_signed,
                "insecure": self.tls_insecure,
                "ca": self.tls_ca,
                "server_cert": self.ssl_cert,
                "client_cert": self.mtls_cert,
                "client_key": self.mtls_key,
                "cert_reqs": repr(self.ssl_cert),
            },
        }

    def connect(self: mqtt_client, keep_alive=None):
        if not keep_alive:
            keep_alive = 60
        else:
            keep_alive = int(keep_alive)
        # g.logger.debug(f"MQTT OPTIONS {self.get_options()=}")
        try:
            if self.tls_ca:
                if self.mtls_key and self.mtls_cert:
                    self.client_id = f"{self.client_id}mTLS-{id_generator()}"
                    self.client = mqtt_client.Client(self.client_id, clean_session=True)
                    self.client.tls_set(
                        ca_certs=self.tls_ca,
                        certfile=self.mtls_cert,
                        keyfile=self.mtls_key,
                        cert_reqs=self.ssl_cert,
                        # tls_version=ssl.PROTOCOL_TLSv1_2
                    )
                    if self.tls_insecure:
                        self.client.tls_insecure_set(True)  # verify CN (COMMON NAME) in certificates
                    g.logger.debug(
                        f"mqtt:connect: '{self.client_id}' ->  '{self.broker}:{self.port}' trying mTLS "
                        f"({'TLS Secure' if not self.tls_insecure else 'TLS Insecure'}) -> tls_ca: "
                        f"'{self.tls_ca}' tls_client_key: '{self.mtls_key}' tls_client_cert: '{self.mtls_cert}'"
                    )

                elif (self.mtls_cert and not self.mtls_key) or (not self.mtls_cert and self.mtls_key):
                    g.logger.debug(
                        f"mqtt:connect:ERROR using mTLS so trying  {self.client_id} -> TLS "
                        f"({'TLS Secure' if not self.tls_insecure else 'TLS Insecure'}) -> tls_ca: "
                        f"{self.tls_ca} tls_client_key: {self.mtls_key} tls_client_cert: {self.mtls_cert}"
                    )
                    self.client_id = f"{self.client_id}TLS-{id_generator()}"
                    self.client = mqtt_client.Client(self.client_id)
                    self.client.tls_set(self.tls_ca, cert_reqs=self.ssl_cert)
                    # ssl.CERT_NONE allows self signed, don't use if using lets encrypt certs and CA
                    if self.tls_insecure:
                        self.client.tls_insecure_set(
                            True
                        )  # DO NOT verify CN (COMMON NAME) in certificates - [MITM risk]

                else:
                    self.client_id = f"{self.client_id}TLS-{id_generator()}"
                    self.client = mqtt_client.Client(self.client_id)
                    self.client.tls_set(self.tls_ca, cert_reqs=ssl.CERT_NONE)
                    g.logger.debug(
                        f"mqtt:connect: {self.client_id} -> {self.broker}:{self.port} trying TLS "
                        f"({'TLS Secure' if not self.tls_insecure else 'TLS Insecure'}) -> tls_ca: {self.tls_ca}"
                    )
            else:
                self.client_id = f"{self.client_id}noTLS-{id_generator()}"
                self.client = mqtt_client.Client(self.client_id)
                show_broker = f"{g.config['sanitize_str']}"
                g.logger.debug(
                    f"mqtt:connect: {self.client_id} -> "
                    f"{self.broker if not str2bool(g.config['sanitize_logs']) else show_broker}:{self.port} "
                    f"{'user:{}'.format(self.user) if self.user else ''} "
                    f"{'passwd:{}'.format(g.config['sanitize_str']) if self.password and self.user else 'passwd:<None>'}"
                )

            if self.user and self.password:
                self.client.username_pw_set(self.user, password=self.password)  # set username and password
            self.client.connect_async(self.broker, port=self.port, keepalive=keep_alive)  # connect to broker
            self.client.loop_start()  # start the loop
            self.client.on_connect = on_connect  # attach function to callback
            # self.client.on_log = on_log
            # connack_string(connack_code)
            # self.client.on_publish = on_publish
            # self.client.on_message=on_message
        except Exception as e:
            g.logger.error(f"mqtt:connect:err_msg-> {e}")
            return print(traceback.format_exc())

        if not self.client:
            g.logger.error(
                f"mqtt:connect: STRANGE ERROR -> there is no active mqtt object instantiated?! Exiting mqtt routine"
            )
            return
        self.conn_wait = 5 if not self.conn_wait else self.conn_wait
        g.logger.debug(2, f"mqtt:connect: connecting to broker (timeout: {self.conn_wait})")
        start = datetime.now()
        while not Connected:  # Wait for connection
            elapsed = datetime.now() - start  # how long has it been
            if elapsed.total_seconds() > self.conn_wait:
                g.logger.error(
                    f"mqtt:connect: broker @ '{self.broker}' did not reply within '{self.conn_wait}' seconds"
                )
                break  # no longer than x seconds waiting for it to connect
        if not Connected:
            g.logger.error(f"mqtt:connect: could not establish a connection to the broker!")
        else:
            self.conn_time = datetime.now()
            self.connected = Connected

    def publish(self, topic=None, message=None, qos=0, retain: bool = False):
        global wasConnected
        if not Connected:
            if wasConnected:
                g.logger.error(f"mqtt:publish: no active connection, attempting to re connect...")
                self.client.reconnect()
                wasConnected = False
            else:
                g.logger.error(f"mqtt:publish: no active connection!")
                return
        if retain:
            self.retain = retain
        self.connected = Connected
        if not message and self.image is not None:
            message = self.image
        if not message:
            g.logger.debug(f"mqtt:publish: no message specified, sending empty message!!")
        if not topic:
            g.logger.error(f"mqtt:publish: no topic specified, please set a topic, skipping publish...")
            return
        if isinstance(message, bytes):
            g.logger.debug(
                2,
                f"mqtt:publish: sending -> topic: '{topic}'  data: '<serialized byte object>'  size: "
                f"{round(message.__sizeof__() / 1024 / 1024, 2)} MB",
            )
        elif not isinstance(message, bytearray):
            g.logger.debug(2, f"mqtt:publish: sending -> topic: '{topic}' data: {message[:255]}")
        else:
            g.logger.debug(
                2,
                f"mqtt:publish: sending -> topic: '{topic}'  data: '<serialized bytearray>'  size: "
                f"{round(message.__sizeof__() / 1024 / 1024, 2)} MB",
            )
        try:
            self.client.publish(topic, message, qos=qos, retain=self.retain)
        except Exception as e:  # todo narrow down exception catching
            return g.logger.error(f"mqtt:publish:err_msg-> {e}")

    def close(self):
        global Connected, wasConnected
        if not Connected:
            return
        try:
            if self.conn_time:
                self.conn_time = datetime.now() - self.conn_time
            show_broker = f"{g.config['sanitize_str']}"
            g.logger.debug(
                2,
                f"mqtt:close: {self.client_id} ->  disconnecting from mqtt broker: "
                f"'{self.broker if not str2bool(g.config['sanitize_logs']) else show_broker}:{self.port}'"
                f" [connection alive for: {self.conn_time.total_seconds()} seconds]",
            )
            self.client.disconnect()
            self.client.loop_stop()
            Connected = self.connected = wasConnected = False
        except Exception as e:
            return g.logger.error(f"mqtt:close:err_msg-> {e}")
