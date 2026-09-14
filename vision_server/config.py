"""Vision server configuration."""

import os

OLLAMA_VLM_HOST = os.getenv("OLLAMA_VLM_HOST", "http://localhost:11434")
OLLAMA_REASONING_HOST = os.getenv("OLLAMA_REASONING_HOST", "http://10.0.0.168:11434")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
REASONING_MODEL = os.getenv("REASONING_MODEL", "qwen2.5:32b")
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
API_KEY = os.getenv("VISION_API_KEY", "wildlife-vision-secret-2026")
USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"
CERT_DIR = os.path.join(os.path.dirname(__file__), "certs")
