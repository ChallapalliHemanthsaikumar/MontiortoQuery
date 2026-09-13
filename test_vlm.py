"""Test Qwen2.5-VL on wildlife camera images."""

import ollama
import time
import sys
import os
import glob


def describe_image(image_path):
    """Send an image to Qwen2.5-VL and get a description."""
    start = time.time()
    response = ollama.chat(
        model='qwen2.5vl:3b',
        messages=[{
            'role': 'user',
            'content': (
                'You are a wildlife monitoring AI. Describe what you see:\n'
                '1. Who/what is in the scene (person, animal, species if possible)\n'
                '2. What are they doing (walking, sitting, eating, etc.)\n'
                '3. Location details (indoor/outdoor, near door, at desk, etc.)\n'
                '4. Any notable details\n'
                'Be concise — 2-3 sentences max.'
            ),
            'images': [image_path]
        }]
    )
    elapsed = time.time() - start
    return response['message']['content'], elapsed


def main():
    # Find all wildlife images
    images = sorted(glob.glob('*.jpg')) + sorted(glob.glob('data/wildlife/*/wildlife_motion/*.jpg'))[:3]
    images += sorted(glob.glob('data/wildlife/*/wildlife_person/*.jpg'))[:3]
    images += sorted(glob.glob('data/wildlife/*/heartbeat/*.jpg'))[:2]

    if not images:
        print("No images found. Place .jpg files in the project directory.")
        return

    print(f"Found {len(images)} images. Analyzing with Qwen2.5-VL...\n")
    print("=" * 60)

    for img in images:
        print(f"\nImage: {img}")
        print("-" * 60)
        description, elapsed = describe_image(img)
        print(f"Description: {description}")
        print(f"Time: {elapsed:.1f}s")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Done. VLM is ready for the vision server pipeline.")


if __name__ == "__main__":
    main()
