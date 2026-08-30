"""Upload captured frames to S3 when motion or ants are detected."""

import cv2
import os
import io
import json
from datetime import datetime, timezone

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class S3ImageUploader:
    """Upload annotated frames to S3."""

    def __init__(self, bucket="antvision-data-dev", experiment_id="exp001"):
        if not HAS_BOTO3:
            raise RuntimeError("boto3 not installed. Run: pip install boto3")
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.experiment_id = experiment_id

    def upload_frame(self, frame, frame_num, trigger="motion", metadata=None):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"{self.experiment_id}/frames/{trigger}_{timestamp}_f{frame_num}.jpg"

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
        print(f"  [upload] {key}")
        return key


class LocalImageSaver:
    """Fallback: save frames locally when S3 is not available."""

    def __init__(self, output_dir="data/captures", experiment_id="exp001"):
        self.output_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def upload_frame(self, frame, frame_num, trigger="motion", metadata=None):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"{trigger}_{timestamp}_f{frame_num}.jpg"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  [saved] {path}")
        return path
