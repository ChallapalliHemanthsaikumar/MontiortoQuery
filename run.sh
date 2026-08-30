#!/bin/bash
cd ~/antvision
PYTHONPATH=. python edge/main.py --live \
  --iot-endpoint a3ix2ylhjo5lpv-ats.iot.us-west-2.amazonaws.com \
  --s3-bucket antvision-data-dev \
  --credentials-endpoint c1bpab6u4qgyi.credentials.iot.us-west-2.amazonaws.com \
  --role-alias antvision-device-alias \
  -o data/test/live_output.avi
