"""WildlifeCam — smart backyard wildlife monitoring on Raspberry Pi.

Captures wildlife activity during daylight hours with intelligent motion
filtering to ignore leaves, wind, and sun changes. Designed to run
unattended for days without filling the SD card.
"""

import argparse
import cv2
import signal
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wildlife.smart_motion import WildlifeMotionDetector
from wildlife.daylight import DaylightDetector
from wildlife.storage_manager import StorageManager
from wildlife.data_logger import DataLogger
from edge.image_uploader import S3ImageUploader, LocalImageSaver

running = True


def handle_signal(sig, frame):
    global running
    print("\nStopping WildlifeCam gracefully...")
    running = False


signal.signal(signal.SIGINT, handle_signal)


def get_camera(args):
    if args.live:
        from edge.camera.pi_camera import PiCamera
        return PiCamera(resolution=(640, 480), fps=args.fps)
    from edge.camera.capture import VideoCapture
    return VideoCapture(args.input)


def get_uploader(args):
    if args.s3_bucket:
        return S3ImageUploader(
            bucket=args.s3_bucket,
            experiment_id=args.experiment,
            credentials_endpoint=args.credentials_endpoint,
            role_alias=args.role_alias,
            cert_dir=args.cert_dir,
        )
    return LocalImageSaver(
        output_dir=os.path.join("data", "wildlife"),
        experiment_id=args.experiment,
    )


def annotate_frame(frame, brightness, motion_pct, boxes, trigger, frame_num,
                   storage_stats):
    annotated = frame.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, "MOTION", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    y_pos = 30
    cv2.putText(annotated, f"WildlifeCam | Frame {frame_num}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += 25
    cv2.putText(annotated, f"Brightness: {brightness}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 22
    cv2.putText(annotated, f"Motion: {motion_pct}%", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 165, 255) if motion_pct > 0.5 else (200, 200, 200), 1)
    y_pos += 22
    if trigger:
        cv2.putText(annotated, f"TRIGGER: {trigger}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_pos += 22
    cv2.putText(annotated,
                f"Disk: {storage_stats['free_gb']}GB | "
                f"Saved: {storage_stats['local_images']} | "
                f"This hr: {storage_stats['captures_this_hour']}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return annotated


def run_wildlife_cam(args):
    capture_interval = args.capture_interval
    heartbeat_interval = args.heartbeat_minutes * 60

    motion_detector = WildlifeMotionDetector(
        min_contour_area=args.min_area,
        min_motion_pct=args.min_motion_pct,
        cooldown_seconds=args.cooldown,
        fps=1.0 / capture_interval,
    )
    daylight = DaylightDetector(brightness_threshold=args.dark_threshold)
    data_dir = os.path.join("data", "wildlife", args.experiment)
    storage = StorageManager(
        data_dir=data_dir,
        min_free_gb=args.min_free_gb,
        max_captures_per_hour=args.max_per_hour,
        max_local_images=args.max_local_images,
    )
    logger = DataLogger(data_dir=data_dir, experiment_id=args.experiment)
    uploader = get_uploader(args)
    camera = get_camera(args)

    print("=" * 50)
    print("  WILDLIFE CAM")
    print("=" * 50)
    print(f"  Experiment:       {args.experiment}")
    print(f"  Capture interval: {capture_interval}s")
    print(f"  Heartbeat:        every {args.heartbeat_minutes} min")
    print(f"  Min motion area:  {args.min_area}px")
    print(f"  Min free disk:    {args.min_free_gb}GB")
    print(f"  Max saves/hour:   {args.max_per_hour}")
    print(f"  Dark threshold:   {args.dark_threshold}")
    print(f"  Mode:             {'live Pi Camera' if args.live else 'video file'}")
    print("=" * 50)

    with camera as cam:
        frame_num = 0
        last_heartbeat = 0
        night_mode = False

        while running:
            loop_start = time.time()
            frame = cam.read()
            if frame is None:
                if args.live:
                    time.sleep(1)
                    continue
                break

            frame_num += 1
            logger.log_frame()
            brightness = daylight.get_brightness(frame)

            if not daylight.is_daylight(frame):
                if not night_mode:
                    print(f"[{frame_num}] Too dark (brightness={brightness}) — "
                          f"pausing captures, checking every 60s")
                    night_mode = True
                logger.log_skip("dark")
                if args.live:
                    time.sleep(60)
                continue

            if night_mode:
                print(f"[{frame_num}] Daylight restored (brightness={brightness}) — resuming")
                night_mode = False

            motion_triggered, motion_pct, boxes = motion_detector.detect(frame)
            should_capture = False
            trigger = ""

            if motion_triggered and not motion_detector.warmup_needed():
                should_capture = True
                trigger = "wildlife_motion"
                logger.log_motion()
                print(f"[{frame_num}] Motion detected: {motion_pct}% "
                      f"({len(boxes)} regions) brightness={brightness}")

            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                should_capture = True
                trigger = trigger or "heartbeat"
                last_heartbeat = now
                logger.log_heartbeat()
                print(f"[{frame_num}] Heartbeat capture — brightness={brightness}")

            if should_capture:
                can_save, reason = storage.can_capture()
                if not can_save:
                    print(f"[{frame_num}] Skipping capture: {reason}")
                    logger.log_skip("storage")
                else:
                    storage_stats = storage.get_stats()
                    annotated = annotate_frame(
                        frame, brightness, motion_pct, boxes, trigger,
                        frame_num, storage_stats
                    )
                    path = uploader.upload_frame(
                        annotated, frame_num, trigger=trigger,
                        metadata={
                            "brightness": str(brightness),
                            "motion_pct": str(motion_pct),
                            "num_detections": str(len(boxes)),
                        }
                    )
                    storage.record_capture()
                    logger.log_event(
                        frame_num, trigger, brightness, motion_pct,
                        len(boxes), boxes, path, storage_stats["free_gb"]
                    )

            if args.show:
                storage_stats = storage.get_stats()
                display = annotate_frame(
                    frame, brightness, motion_pct, boxes, trigger,
                    frame_num, storage_stats
                )
                cv2.imshow("WildlifeCam", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Periodic cleanup
            if frame_num % 100 == 0:
                deleted = storage.cleanup_old_frames()
                if deleted > 0:
                    print(f"[{frame_num}] Cleaned up {deleted} old frames")

            if args.live:
                elapsed = time.time() - loop_start
                sleep_time = max(0, capture_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    if args.show:
        cv2.destroyAllWindows()

    summary_path = logger.save_summary()
    print("\n" + "=" * 50)
    print("  WILDLIFE CAM — SESSION COMPLETE")
    print("=" * 50)
    stats = logger.stats
    print(f"  Total frames:     {stats['total_frames']}")
    print(f"  Images saved:     {stats['images_saved']}")
    print(f"  Motion triggers:  {stats['motion_triggers']}")
    print(f"  Heartbeats:       {stats['heartbeat_captures']}")
    print(f"  Dark skips:       {stats['dark_skips']}")
    print(f"  Storage skips:    {stats['storage_skips']}")
    print(f"  Summary:          {summary_path}")
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="WildlifeCam — Smart Backyard Wildlife Monitor")

    # Source
    parser.add_argument("--live", action="store_true",
                        help="Use Raspberry Pi camera (default: video file)")
    parser.add_argument("--input", "-i", default=None,
                        help="Path to test video file")
    parser.add_argument("--fps", type=float, default=15,
                        help="Pi Camera FPS (default: 15)")
    parser.add_argument("--show", action="store_true",
                        help="Show live preview window")

    # Experiment
    parser.add_argument("--experiment", "-e", default="wildlife001",
                        help="Experiment ID (default: wildlife001)")

    # Capture timing
    parser.add_argument("--capture-interval", type=float, default=5.0,
                        help="Seconds between frame grabs (default: 5)")
    parser.add_argument("--heartbeat-minutes", type=int, default=30,
                        help="Minutes between heartbeat captures (default: 30)")
    parser.add_argument("--cooldown", type=float, default=15.0,
                        help="Seconds cooldown after a motion trigger (default: 15)")

    # Motion sensitivity
    parser.add_argument("--min-area", type=int, default=3000,
                        help="Min contour area in pixels to count as wildlife (default: 3000)")
    parser.add_argument("--min-motion-pct", type=float, default=0.5,
                        help="Min percent of frame with motion (default: 0.5)")

    # Daylight
    parser.add_argument("--dark-threshold", type=int, default=30,
                        help="Brightness below this = too dark (default: 30)")

    # Storage
    parser.add_argument("--min-free-gb", type=float, default=5.0,
                        help="Stop capturing if disk free < this (default: 5.0)")
    parser.add_argument("--max-per-hour", type=int, default=120,
                        help="Max captures per hour (default: 120)")
    parser.add_argument("--max-local-images", type=int, default=2000,
                        help="Auto-cleanup oldest frames above this count (default: 2000)")

    # Cloud upload
    parser.add_argument("--s3-bucket", default=None,
                        help="S3 bucket for uploads")
    parser.add_argument("--credentials-endpoint", default=None)
    parser.add_argument("--role-alias", default="antvision-device-alias")
    parser.add_argument("--cert-dir", default="certs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_wildlife_cam(args)
