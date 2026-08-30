#!/bin/bash
# Load config from .env file (not committed to git)
source ~/antvision/.env

cd ~/antvision
PYTHONPATH=. python edge/main.py --live \
  --iot-endpoint "$IOT_ENDPOINT" \
  --s3-bucket "$S3_BUCKET" \
  --credentials-endpoint "$CREDENTIALS_ENDPOINT" \
  --role-alias "$ROLE_ALIAS" \
  -o data/test/live_output.avi
