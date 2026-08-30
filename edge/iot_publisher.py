"""Publish behavioral events to AWS IoT Core via MQTT."""

import json
import os

try:
    from awscrt import mqtt
    from awsiot import mqtt_connection_builder
    HAS_IOT_SDK = True
except ImportError:
    HAS_IOT_SDK = False


class IoTPublisher:
    """Send events to AWS IoT Core MQTT topic."""

    def __init__(self, endpoint, cert_dir="certs", client_id="antvision-pi01",
                 topic="antvision/events"):
        if not HAS_IOT_SDK:
            raise RuntimeError("awsiotsdk not installed. Run: pip install awsiotsdk")

        self.topic = topic
        self.connection = mqtt_connection_builder.mtls_from_path(
            endpoint=endpoint,
            cert_filepath=os.path.join(cert_dir, "device-certificate.pem.crt"),
            pri_key_filepath=os.path.join(cert_dir, "private.pem.key"),
            ca_filepath=os.path.join(cert_dir, "AmazonRootCA1.pem"),
            client_id=client_id,
            clean_session=False,
            keep_alive_secs=30,
        )

    def connect(self):
        future = self.connection.connect()
        future.result()
        print(f"Connected to AWS IoT Core")
        return self

    def publish(self, event):
        payload = json.dumps(event)
        self.connection.publish(
            topic=self.topic,
            payload=payload,
            qos=mqtt.QoS.AT_LEAST_ONCE,
        )

    def disconnect(self):
        future = self.connection.disconnect()
        future.result()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.disconnect()


class LocalPublisher:
    """Fallback publisher that logs events locally (for development)."""

    def __init__(self):
        self.events = []

    def connect(self):
        print("Using local publisher (no AWS connection)")
        return self

    def publish(self, event):
        self.events.append(event)

    def disconnect(self):
        pass

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.disconnect()
