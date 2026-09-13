"""Qwen2.5-VL wrapper via Ollama."""

import base64
import ollama
from config import VLM_MODEL

SYSTEM_PROMPT = (
    "You are a wildlife and activity monitoring AI. "
    "Describe what you see concisely in 2-3 sentences. Include: "
    "1) Who/what is in the scene (person, animal species if possible) "
    "2) What they are doing "
    "3) Location context. "
    "Do not mention camera overlays or text on the image."
)


def describe_frame(image_bytes, yolo_class="", yolo_confidence=0.0):
    """Run Qwen2.5-VL on raw image bytes, return description."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    hint = ""
    if yolo_class:
        hint = f" YOLO pre-detection: {yolo_class} ({yolo_confidence:.0%} confidence)."

    response = ollama.chat(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Describe this wildlife camera frame.{hint}",
                "images": [b64_image],
            },
        ],
    )
    return response["message"]["content"]
