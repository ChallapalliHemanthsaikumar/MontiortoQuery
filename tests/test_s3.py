from edge.image_uploader import S3ImageUploader
import numpy as np

uploader = S3ImageUploader(
    bucket="antvision-data-dev",
    experiment_id="exp001",
    credentials_endpoint="c1bpab6u4qgyi.credentials.iot.us-west-2.amazonaws.com",
    role_alias="antvision-device-alias",
    cert_dir="certs",
)
print("S3 uploader created OK")

test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
uploader.upload_frame(test_frame, 0, trigger="test")
print("Upload successful!")
