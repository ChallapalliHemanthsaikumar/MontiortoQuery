"""Generate timelapse video from heartbeat captures."""

import os
import cv2
import glob


def generate_timelapse(image_dir, output_path="data/wildlife/timelapse.mp4",
                       fps=10, resolution=(640, 480)):
    """Stitch images into a timelapse video.

    Args:
        image_dir: directory containing captured images
        output_path: where to save the video
        fps: playback frames per second
        resolution: output video resolution
    """
    patterns = [
        os.path.join(image_dir, "heartbeat", "*.jpg"),
        os.path.join(image_dir, "**", "heartbeat", "*.jpg"),
    ]

    images = []
    for pat in patterns:
        images.extend(glob.glob(pat, recursive=True))
    images = sorted(set(images))

    if not images:
        all_jpgs = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"),
                                    recursive=True))
        if all_jpgs:
            print(f"No heartbeat images found. Using all {len(all_jpgs)} images.")
            images = all_jpgs

    if not images:
        print("No images found for timelapse.")
        return None

    print(f"Creating timelapse from {len(images)} images at {fps} fps...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame = cv2.resize(frame, resolution)

        ts = os.path.basename(img_path).split("_f")[0]
        cv2.putText(frame, ts, (10, resolution[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(images)}] frames written")

    writer.release()

    duration = len(images) / fps
    print(f"\nTimelapse saved: {output_path}")
    print(f"  {len(images)} frames, {fps} fps, {duration:.1f}s playback")

    return output_path
