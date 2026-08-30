"""Upload captured frames to S3 using IoT credential provider (no stored keys)."""

import cv2
import os
import json
from datetime import datetime, timezone

try:
    import boto3
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    try:
        import urllib.request
        import ssl
        HAS_REQUESTS = False
    except ImportError:
        HAS_REQUESTS = False


def _get_iot_credentials(endpoint, role_alias, cert_path, key_path, ca_path):
    """Get temporary AWS credentials using IoT certificate."""
    url = f"https://{endpoint}/role-aliases/{role_alias}/credentials"

    if HAS_REQUESTS:
        resp = _requests.get(url, cert=(cert_path, key_path), verify=ca_path)
        return resp.json()["credentials"]
    else:
        ctx = ssl.create_default_context(cafile=ca_path)
        ctx.load_cert_chain(cert_path, key_path)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx) as resp:
            return json.loads(resp.read())["credentials"]


def _make_refreshable_session(endpoint, role_alias, cert_path, key_path, ca_path):
    """Create a botocore session with auto-refreshing IoT credentials."""

    def _refresh():
        creds = _get_iot_credentials(endpoint, role_alias, cert_path, key_path, ca_path)
        return {
            "access_key": creds["accessKeyId"],
            "secret_key": creds["secretAccessKey"],
            "token": creds["sessionToken"],
            "expiry_time": creds["expiration"],
        }

    refreshable = RefreshableCredentials.create_from_metadata(
        metadata=_refresh(),
        refresh_using=_refresh,
        method="sts-assume-role",
    )

    session = get_session()
    session._credentials = refreshable
    return boto3.Session(botocore_session=session)


class S3ImageUploader:
    """Upload annotated frames to S3 using IoT credentials (no stored keys)."""

    def __init__(self, bucket="antvision-data-dev", experiment_id="exp001",
                 credentials_endpoint=None, role_alias=None, cert_dir="certs"):
        if not HAS_BOTO3:
            raise RuntimeError("boto3 not installed. Run: pip install boto3")

        self.bucket = bucket
        self.experiment_id = experiment_id

        if credentials_endpoint and role_alias:
            session = _make_refreshable_session(
                endpoint=credentials_endpoint,
                role_alias=role_alias,
                cert_path=os.path.join(cert_dir, "device-certificate.pem.crt"),
                key_path=os.path.join(cert_dir, "private.pem.key"),
                ca_path=os.path.join(cert_dir, "AmazonRootCA1.pem"),
            )
            self.s3 = session.client("s3")
        else:
            self.s3 = boto3.client("s3")

    def upload_frame(self, frame, frame_num, trigger="motion", metadata=None):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"{self.experiment_id}/{trigger}/{timestamp}_f{frame_num}.jpg"

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        extra = metadata or {}
        extra["frame_num"] = str(frame_num)
        extra["trigger"] = trigger
        extra["timestamp"] = timestamp

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.tobytes(),
            ContentType="image/jpeg",
            Metadata=extra,
        )
        print(f"  [upload] s3://{self.bucket}/{key}")
        return key


class LocalImageSaver:
    """Fallback: save frames locally when S3 is not available."""

    def __init__(self, output_dir="data/captures", experiment_id="exp001"):
        self.output_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def upload_frame(self, frame, frame_num, trigger="motion", metadata=None):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        trigger_dir = os.path.join(self.output_dir, trigger)
        os.makedirs(trigger_dir, exist_ok=True)
        filename = f"{timestamp}_f{frame_num}.jpg"
        path = os.path.join(trigger_dir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  [saved] {path}")
        return path
