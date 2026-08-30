"""AntVision edge pipeline — detect, track, and analyze ant behavior."""

import argparse
import cv2
import signal
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edge.camera.capture import VideoCapture
from edge.detection.preprocessor import Preprocessor
from edge.detection.detector import AntDetector
from edge.detection.food_zone import FoodZone
from edge.detection.motion import MotionDetector
from edge.tracking.tracker import CentroidTracker
from edge.tracking.trajectory import TrajectoryRecorder
from edge.analytics import BehaviorAnalyzer
from edge.event_emitter import EventEmitter
from edge.iot_publisher import LocalPublisher
from edge.image_uploader import LocalImageSaver

SNAPSHOT_INTERVAL = 15
running = True


def handle_signal(sig, frame):
    global running
    print("\nStopping pipeline gracefully...")
    running = False


signal.signal(signal.SIGINT, handle_signal)


def get_capture(args):
    if args.live:
        from edge.camera.pi_camera import PiCamera
        return PiCamera(resolution=(640, 480), fps=15)
    return VideoCapture(args.input)


def get_publisher(args):
    if args.iot_endpoint:
        from edge.iot_publisher import IoTPublisher
        return IoTPublisher(
            endpoint=args.iot_endpoint,
            cert_dir=args.cert_dir,
        )
    return LocalPublisher()


def get_uploader(args):
    if args.s3_bucket:
        from edge.image_uploader import S3ImageUploader
        return S3ImageUploader(
            bucket=args.s3_bucket,
            experiment_id=args.experiment,
            credentials_endpoint=args.credentials_endpoint,
            role_alias=args.role_alias,
            cert_dir=args.cert_dir,
        )
    return LocalImageSaver(experiment_id=args.experiment)


def run_pipeline(args):
    preprocessor = Preprocessor()
    detector = AntDetector()
    tracker = CentroidTracker()
    motion_detector = MotionDetector(cooldown_frames=int(args.capture_interval))
    emitter = EventEmitter(experiment_id=args.experiment)

    capture = get_capture(args)
    publisher = get_publisher(args)
    uploader = get_uploader(args)

    with capture as cap, publisher as pub:
        fps = getattr(cap, 'fps', 15.0) or 15.0

        trajectory = TrajectoryRecorder(fps=fps)
        analyzer = BehaviorAnalyzer(fps=fps)
        food_zone = FoodZone(x=360, y=300, w=120, h=120)

        emitter.emit_experiment_start()
        writer = None
        frame_num = 0
        prev_ant_count = 0

        while running:
            frame = cap.read()
            if frame is None:
                break

            mask = preprocessor.process(frame)
            detections = detector.detect(mask)
            tracked = tracker.update(detections)
            ant_count = len(tracked)

            trajectory.update(tracked, frame_num)
            transitions = food_zone.update(tracked, frame_num)
            analyzer.record_transitions(transitions)
            analyzer.record_frame(len(detections), len(food_zone.inside))

            motion_detected, motion_pct = motion_detector.detect(frame)

            # Build annotated frame
            annotated = detector.draw_detections(frame, detections)
            annotated = trajectory.draw_trails(annotated)
            annotated = food_zone.draw(annotated)

            for obj_id, centroid in tracked.items():
                cx, cy = int(centroid[0]), int(centroid[1])
                speed = trajectory.get_speed(obj_id)
                cv2.putText(annotated, f"ID {obj_id} ({speed:.0f}px/s)",
                            (cx - 10, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

            cv2.putText(annotated, f"Tracked: {ant_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(annotated, f"Frame: {frame_num}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if motion_pct > 0:
                cv2.putText(annotated, f"Motion: {motion_pct}%", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # Upload frame on motion, ant activity, or periodic heartbeat
            should_capture = False
            trigger = ""

            periodic_interval = int(fps * 30)
            if motion_detected:
                should_capture = True
                trigger = "motion"
            if ant_count > 0 and prev_ant_count == 0:
                should_capture = True
                trigger = "ant_arrival"
            if len(transitions) > 0:
                should_capture = True
                trigger = "zone_event"
            if periodic_interval > 0 and frame_num % periodic_interval == 0:
                should_capture = True
                trigger = trigger or "heartbeat"

            if should_capture:
                uploader.upload_frame(annotated, frame_num, trigger=trigger,
                                      metadata={"ant_count": str(ant_count),
                                                 "motion_pct": str(motion_pct)})

            # Publish zone events
            for t in transitions:
                speed = trajectory.get_speed(t.ant_id)
                if t.event.value == "food_zone_enter":
                    event = emitter.emit_zone_enter(t.ant_id, t.frame, t.x, t.y, speed)
                else:
                    event = emitter.emit_zone_exit(t.ant_id, t.frame, t.x, t.y)
                pub.publish(event)
                print(f"  [{t.event.value}] Ant {t.ant_id} at frame {t.frame}")

            # Periodic metrics snapshot
            if frame_num % SNAPSHOT_INTERVAL == 0:
                speeds = trajectory.get_all_speeds()
                avg_speed = sum(speeds.values()) / len(speeds) if speeds else 0.0
                event = emitter.emit_metrics_snapshot(
                    frame_num, len(detections), len(food_zone.inside), avg_speed
                )
                pub.publish(event)

            # Write video
            if args.output and writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

            if writer:
                writer.write(annotated)

            if args.show:
                cv2.imshow("AntVision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_ant_count = ant_count
            frame_num += 1

    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    emitter.emit_experiment_end(frame_num)

    metrics = analyzer.compute_metrics(tracker, trajectory)

    print("\n" + "=" * 50)
    print("  ANTVISION EXPERIMENT RESULTS")
    print("=" * 50)
    print(metrics.to_json())
    print("=" * 50)

    out_dir = os.path.dirname(args.output or args.input or "data/test/")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        f.write(metrics.to_json())

    events_path = os.path.join(out_dir, "events.json")
    count = emitter.save_events(events_path)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Events saved to {events_path} ({count} events)")


def parse_args():
    parser = argparse.ArgumentParser(description="AntVision Edge Pipeline")
    parser.add_argument("--input", "-i", default="data/test/test_ants.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", "-o", default="data/test/output_tracked.avi",
                        help="Path to output annotated video")
    parser.add_argument("--live", action="store_true",
                        help="Use Raspberry Pi camera instead of video file")
    parser.add_argument("--show", action="store_true",
                        help="Show live preview window")
    parser.add_argument("--experiment", "-e", default="exp001",
                        help="Experiment ID")
    parser.add_argument("--iot-endpoint", default=None,
                        help="AWS IoT Core endpoint for publishing events")
    parser.add_argument("--cert-dir", default="certs",
                        help="Directory containing IoT certificates")
    parser.add_argument("--s3-bucket", default=None,
                        help="S3 bucket for image uploads")
    parser.add_argument("--credentials-endpoint", default=None,
                        help="IoT credentials provider endpoint for S3 access")
    parser.add_argument("--role-alias", default="antvision-device-alias",
                        help="IoT role alias for credentials provider")
    parser.add_argument("--capture-interval", default=15, type=int,
                        help="Min frames between motion captures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
