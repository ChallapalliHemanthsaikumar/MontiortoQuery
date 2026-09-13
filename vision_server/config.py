"""Vision server configuration."""

import os

VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:3b")
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
API_KEY = os.getenv("VISION_API_KEY", "wildlife-vision-secret-2026")
USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"
CERT_DIR = os.path.join(os.path.dirname(__file__), "certs")
