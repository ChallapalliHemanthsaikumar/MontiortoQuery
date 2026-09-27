"""Multimodal Activity Monitor V2 — Edge Pipeline.

This runs on the Raspberry Pi. It captures video + audio simultaneously,
runs motion detection + YOLO + YAMNet, fuses the signals, deduplicates,
records DYNAMIC-LENGTH A/V clips, and pushes events to the GPU server.

Dynamic recording:
    Trigger fires → start recording (drain ring buffer for pre-roll)
    Trigger active → keep recording (feed frames in real-time)
    Trigger stops → post-roll countdown (5s default)
    Post-roll expires → finalize clip → push to server

    One event = one clip, however long it takes.
    A 3-minute conversation = one 3-minute clip, not 12 fixed clips.

Rolling storage:
    Clips stay on Pi for 24+ hours within a 3GB budget.
    When budget is exceeded, oldest clips are deleted first.
    Pi never runs out of disk space.

Usage (Pi, live):
    python -m edge_v2.main --live --device /dev/video2 --mic plughw:3,0 \
        --vision-server http://10.0.0.181:8000

Usage (Windows/Mac, test with webcam):
    python -m edge_v2.main --live --show

Usage (test with video file, no audio):
    python -m edge_v2.main --input data/test/test_clip.mp4 --no-audio
"""

import argparse
import cv2
import signal
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edge_v2.ring_buffer import VideoRingBuffer, AudioRingBuffer
from edge_v2.fusion_gate import FusionGate, FusionResult, Decision, EventType
from edge_v2.dedup import SmartDedup
from edge_v2.clip_recorder import DynamicClipRecorder, RollingStorage

running = True


def handle_signal(sig, frame):
    global running
    print("\nStopping V2 pipeline gracefully...")
    running = False


signal.signal(signal.SIGINT, handle_signal)


def get_camera(args):
    if args.live:
        device = args.device
        if device is None:
            device = 0
        elif device.isdigit():
            device = int(device)
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        return cap
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")
    return cap


def load_yolo(args):
    model_path = os.path.join("models", "yolov8n.onnx")
    if not os.path.exists(model_path):
        model_path = "yolov8n.onnx"
    if not os.path.exists(model_path):
        print("  [WARN] yolov8n.onnx not found. YOLO disabled.")
        return None

    net = cv2.dnn.readNetFromONNX(model_path)
    print(f"  YOLOv8n ONNX loaded from {model_path}")
    return net


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

WILDLIFE_SET = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}


def run_yolo_onnx(net, frame, confidence=0.35):
    if net is None:
        return []

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    output = outputs[0].transpose(1, 0)

    h, w = frame.shape[:2]
    x_scale = w / 640
    y_scale = h / 640

    detections = []
    for row in output:
        cx, cy, bw, bh = row[:4]
        scores = row[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])

        if conf < confidence:
            continue

        x1 = max(0, int((cx - bw / 2) * x_scale))
        y1 = max(0, int((cy - bh / 2) * y_scale))
        x2 = min(w, int((cx + bw / 2) * x_scale))
        y2 = min(h, int((cy + bh / 2) * y_scale))

        name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"

        detections.append({
            "class_id": class_id,
            "class_name": name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "is_wildlife": name in WILDLIFE_SET,
        })

    if detections:
        boxes = np.array([d["bbox"] for d in detections])
        confs = np.array([d["confidence"] for d in detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(),
                                    confidence, 0.45)
        if len(indices) > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices]

    return detections


def annotate_frame(frame, motion_pct, boxes, yolo_detections,
                   audio_results, fusion_result, frame_num,
                   recorder_state, recorder_duration, storage_stats):
    annotated = frame.copy()
    h_frame, w_frame = annotated.shape[:2]

    # --- Motion bounding boxes (green, thin) ---
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 1)

    # --- YOLO bounding boxes (thick, with labels) ---
    # Person = red, animal = orange, vehicle = cyan, other = yellow
    if yolo_detections:
        for det in yolo_detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["class_name"]
            conf = det["confidence"]

            if name == "person":
                color = (0, 0, 255)
            elif name in WILDLIFE_SET:
                color = (0, 165, 255)
            elif name in ("car", "truck", "bus", "motorcycle", "bicycle"):
                color = (255, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_w, label_h = label_size
            cv2.rectangle(annotated, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # --- Recording indicator (top-right, prominent) ---
    if recorder_state in ("recording", "post_roll"):
        rec_color = (0, 0, 255) if recorder_state == "recording" else (0, 165, 255)
        rec_label = f"REC {recorder_duration:.1f}s" if recorder_state == "recording" else f"POST-ROLL {recorder_duration:.1f}s"
        cv2.circle(annotated, (w_frame - 20, 25), 8, rec_color, -1)
        cv2.putText(annotated, rec_label, (w_frame - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_color, 2)

    # --- HUD overlay (top-left) ---
    y_pos = 22
    cv2.putText(annotated, f"V2 Monitor | Frame {frame_num}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    y_pos += 20
    motion_color = (0, 165, 255) if motion_pct > 0.5 else (150, 150, 150)
    cv2.putText(annotated, f"Motion: {motion_pct}%", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, motion_color, 1)

    # Object summary line
    if yolo_detections:
        y_pos += 20
        obj_summary = {}
        for d in yolo_detections:
            n = d["class_name"]
            obj_summary[n] = obj_summary.get(n, 0) + 1
        obj_str = " | ".join(f"{n}: {c}" for n, c in obj_summary.items())
        cv2.putText(annotated, f"Objects: {obj_str}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    if audio_results:
        y_pos += 20
        audio_str = " | ".join(f"{n} {c:.0%}" for n, c in audio_results[:2])
        cv2.putText(annotated, f"Audio: {audio_str}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    if fusion_result:
        y_pos += 20
        if fusion_result.decision == Decision.SAVE:
            decision_color = (0, 255, 0)
            decision_str = f"SAVE: {fusion_result.reason}"
        else:
            decision_color = (100, 100, 100)
            decision_str = f"skip: {fusion_result.reason}"
        cv2.putText(annotated, decision_str, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, decision_color, 1)

    # --- Storage stats (bottom bar) ---
    cv2.rectangle(annotated, (0, h_frame - 22), (w_frame, h_frame), (30, 30, 30), -1)
    cv2.putText(annotated,
                f"Clips: {storage_stats['clip_count']} | "
                f"{storage_stats['total_size_mb']}MB / {storage_stats['budget_gb']}GB | "
                f"Oldest: {storage_stats['oldest_hours']}h | "
                f"Disk free: {storage_stats['free_disk_gb']}GB",
                (8, h_frame - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    return annotated


def run_pipeline(args):
    print("=" * 60)
    print("  MULTIMODAL ACTIVITY MONITOR V2")
    print("=" * 60)

    # --- Ring buffers ---
    video_buffer = VideoRingBuffer(duration_seconds=args.pre_roll, fps=args.fps)
    audio_buffer = AudioRingBuffer(
        duration_seconds=args.pre_roll,
        sample_rate=args.sample_rate,
        chunk_size=1024,
    )
    print(f"  Ring buffers: {args.pre_roll}s pre-roll")

    # --- Motion detector (reuse from wildlife module) ---
    from wildlife.smart_motion import WildlifeMotionDetector
    motion_detector = WildlifeMotionDetector(
        min_contour_area=args.min_area,
        min_motion_pct=args.min_motion_pct,
        cooldown_seconds=args.cooldown,
        fps=1.0 / max(args.capture_interval, 0.033),
        min_solidity=args.min_solidity,
        confirm_frames=args.confirm_frames,
        match_distance=args.match_distance,
    )
    print(f"  Motion: area>{args.min_area}px solidity>{args.min_solidity}")

    # --- YOLO ---
    yolo_net = load_yolo(args)

    # --- Audio classifier ---
    # Auto-selects: YAMNet ONNX (if available) → simple spectral (numpy)
    # No tensorflow needed. No heavy dependencies.
    audio_classifier = None
    if not args.no_audio:
        try:
            from edge_v2.audio_classifier import AudioClassifier
            audio_classifier = AudioClassifier()
        except Exception as e:
            print(f"  [WARN] Audio classifier: {e}")

    # --- Audio capture thread ---
    audio_capture = None
    if not args.no_audio and audio_classifier:
        try:
            from edge_v2.audio_capture import AudioCapture
            audio_capture = AudioCapture(
                ring_buffer=audio_buffer,
                sample_rate=args.sample_rate,
                device=args.mic,
                highpass_hz=200,
            )
            audio_capture.start()
            print(f"  Mic: {args.mic or 'default'} @ {args.sample_rate}Hz")
        except Exception as e:
            print(f"  [WARN] Mic: {e} — video-only mode")
            audio_capture = None

    # --- Fusion + dedup ---
    fusion = FusionGate()
    dedup = SmartDedup(phash_threshold=args.phash_threshold)

    # --- Dynamic clip recorder ---
    recorder = DynamicClipRecorder(
        output_dir=args.clip_dir,
        post_roll_seconds=args.post_roll,
        fps=args.fps,
        sample_rate=args.sample_rate,
        max_clip_seconds=args.max_clip_seconds,
    )
    print(f"  Recording: dynamic length, {args.post_roll}s post-roll, "
          f"max {args.max_clip_seconds}s")

    # --- Rolling storage ---
    storage = RollingStorage(
        clip_dir=args.clip_dir,
        max_gb=args.max_storage_gb,
        min_free_sd_gb=args.min_free_gb,
    )
    print(f"  Storage: {args.max_storage_gb}GB rolling budget, "
          f"keep {args.min_free_gb}GB free")

    # --- Event pusher ---
    pusher = None
    if args.vision_server:
        from edge_v2.event_push import EventPusher
        pusher = EventPusher(
            server_url=args.vision_server,
            api_key=args.vision_api_key,
            camera_id=args.camera_id,
        )
        pusher.start()
        print(f"  Push: {args.vision_server}")

    # --- Camera ---
    camera = get_camera(args)

    print(f"  Camera: {'live' if args.live else args.input}")
    print(f"  Mode: {'video+audio' if audio_capture else 'video-only'}")
    print("=" * 60)
    print("  Press Ctrl+C to stop\n")

    frame_num = 0
    last_audio_classify = 0.0
    audio_results = []
    audio_priority = "ignore"
    events_saved = 0
    events_skipped = 0
    last_heartbeat = time.time()
    last_storage_cleanup = time.time()
    storage_stats = storage.get_stats()
    last_fusion_event_type = None

    sticky_yolo_detections = []
    sticky_yolo_age = 0
    YOLO_STICKY_MAX_AGE = 10
    YOLO_RECHECK_INTERVAL = 5
    last_audio_feed_ts = 0.0

    try:
        while running:
            loop_start = time.time()

            ret, frame = camera.read()
            if not ret or frame is None:
                if args.live:
                    time.sleep(0.5)
                    continue
                break

            if args.rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif args.rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frame_num += 1
            now = time.time()

            # Always feed video ring buffer (for pre-roll capture)
            video_buffer.push(frame)

            # --- Audio classification (every 1 second) ---
            if audio_classifier and now - last_audio_classify >= 1.0:
                audio_chunks = audio_buffer.get_last_n_seconds(1.0)
                if audio_chunks:
                    pcm = np.concatenate([chunk for _, chunk in audio_chunks])
                    audio_results = audio_classifier.classify(
                        pcm, source_rate=args.sample_rate
                    )
                    audio_priority = audio_classifier.get_priority(audio_results)
                    last_audio_classify = now

            # --- Motion detection ---
            motion_triggered, motion_pct, boxes = motion_detector.detect(frame)

            # --- YOLO ---
            # Three cases when YOLO runs:
            #   1. Motion triggered (normal detection flow)
            #   2. While recording, every YOLO_RECHECK_INTERVAL frames
            #      (keeps tracking the person/object so trigger stays alive)
            #   3. Never during warmup (background subtractor needs to settle)
            yolo_detections = []
            run_yolo_now = False

            if motion_triggered and not motion_detector.warmup_needed():
                run_yolo_now = True
            elif recorder.is_recording and frame_num % YOLO_RECHECK_INTERVAL == 0:
                run_yolo_now = True

            if run_yolo_now:
                yolo_detections = run_yolo_onnx(
                    yolo_net, frame, args.yolo_confidence
                )

            # Sticky YOLO: keep last detections alive for a few frames.
            # This prevents the trigger from dropping between YOLO runs
            # and keeps bounding boxes visible on the preview.
            if yolo_detections:
                sticky_yolo_detections = yolo_detections
                sticky_yolo_age = 0
            else:
                sticky_yolo_age += 1
                if sticky_yolo_age > YOLO_STICKY_MAX_AGE:
                    sticky_yolo_detections = []

            # Use sticky detections for fusion gate (so trigger stays alive
            # between YOLO runs) and for display
            effective_detections = sticky_yolo_detections

            # --- Fusion gate ---
            fusion_result = fusion.decide(
                motion_triggered=motion_triggered or recorder.is_recording,
                yolo_detections=effective_detections,
                audio_results=audio_results,
                audio_priority=audio_priority,
            )

            # --- Heartbeat override ---
            if now - last_heartbeat >= args.heartbeat_minutes * 60:
                if fusion_result.decision != Decision.SAVE:
                    audio_name = audio_results[0][0] if audio_results else "silence"
                    fusion_result = FusionResult(
                        Decision.SAVE, EventType.HEARTBEAT, 0.0,
                        "nothing", audio_name, audio_priority, "heartbeat"
                    )
                last_heartbeat = now

            # ================================================================
            # DYNAMIC RECORDING STATE MACHINE
            #
            # Not recording + SAVE decision → start recording
            # Recording + SAVE decision     → keep recording (extend clip)
            # Recording + SKIP decision     → trigger ended → post-roll starts
            # Post-roll + SAVE decision     → cancel post-roll, resume
            # Post-roll expires             → finalize → encode → push
            # ================================================================

            trigger_active = fusion_result.decision == Decision.SAVE

            if not recorder.is_recording and trigger_active:
                # --- NEW EVENT: start recording ---
                event_type = (fusion_result.event_type.value
                              if fusion_result.event_type else "unknown")

                should_save, dedup_reason = dedup.check(
                    frame, event_type, fusion_result.audio_class
                )

                if should_save and storage.can_record():
                    pre_video = video_buffer.drain()
                    pre_audio = audio_buffer.drain()

                    recorder.start_recording(pre_video, pre_audio, event_type)
                    last_fusion_event_type = event_type
                    last_audio_feed_ts = time.time()
                    print(f"[{frame_num}] REC START: {event_type} — "
                          f"{fusion_result.reason}")
                else:
                    events_skipped += 1
                    if frame_num % 100 == 0 and not should_save:
                        print(f"[{frame_num}] DEDUP skip: {dedup_reason}")

            elif recorder.is_recording and trigger_active:
                # --- TRIGGER STILL ACTIVE: extend recording ---
                if recorder.state == "post_roll":
                    recorder.trigger_renewed()
                    print(f"[{frame_num}] REC RESUMED (trigger re-activated)")

            elif recorder.is_recording and not trigger_active:
                # --- TRIGGER ENDED: start post-roll countdown ---
                if recorder.state == "recording":
                    recorder.trigger_ended()
                    print(f"[{frame_num}] POST-ROLL started "
                          f"({args.post_roll}s countdown)")

            # Feed current frame/audio to recorder (during recording + post-roll)
            if recorder.is_recording:
                recorder.feed_frame(frame, effective_detections)
                if audio_capture:
                    new_audio = audio_buffer.get_chunks_after(last_audio_feed_ts)
                    for ts, chunk in new_audio:
                        recorder.feed_audio(chunk)
                        last_audio_feed_ts = ts

            # Check if recorder just finished (state went idle after post-roll)
            if recorder.state == "idle" and last_fusion_event_type is not None:
                # Clip just finalized
                clip_path = recorder.get_clip_path()
                best_frame = recorder.get_best_frame()
                duration = recorder.duration

                if clip_path:
                    events_saved += 1
                    dedup.record_save(
                        frame, last_fusion_event_type,
                        fusion_result.audio_class,
                    )
                    print(f"[{frame_num}] REC DONE: {clip_path} "
                          f"({duration:.1f}s)")

                    # Push to server
                    if pusher:
                        audio_pcm = recorder.get_audio_pcm()
                        pusher.push_event(
                            clip_path=clip_path,
                            best_frame=best_frame,
                            audio_pcm=audio_pcm,
                            sample_rate=args.sample_rate,
                            metadata={
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "yolo_class": fusion_result.visual_class,
                                "yolo_confidence": str(fusion_result.confidence),
                                "audio_class": fusion_result.audio_class,
                                "audio_priority": fusion_result.audio_priority,
                                "event_type": last_fusion_event_type,
                                "motion_pct": str(motion_pct),
                                "frame_num": str(frame_num),
                                "duration": str(round(duration, 1)),
                            },
                        )

                    # Cleanup rolling storage
                    deleted = storage.cleanup()
                    if deleted > 0:
                        print(f"  [storage] Deleted {deleted} old clips")
                    storage_stats = storage.get_stats()

                last_fusion_event_type = None

            # --- Periodic storage stats refresh ---
            if now - last_storage_cleanup >= 60:
                storage_stats = storage.get_stats()
                last_storage_cleanup = now

            # --- Live preview ---
            if args.show:
                display = annotate_frame(
                    frame, motion_pct, boxes, effective_detections,
                    audio_results, fusion_result, frame_num,
                    recorder.state, recorder.duration, storage_stats,
                )
                cv2.imshow("V2 Monitor", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # --- Frame rate control ---
            if args.live:
                elapsed = time.time() - loop_start
                sleep_time = max(0, args.capture_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        # Finalize any in-progress recording before shutdown
        if recorder.is_recording:
            clip_path = recorder.get_clip_path()
            recorder.force_stop()
            events_saved += 1
            print(f"  [shutdown] Finalized in-progress clip: {clip_path}")

            if pusher and clip_path:
                best_frame = recorder.get_best_frame()
                audio_pcm = recorder.get_audio_pcm()
                pusher.push_event(
                    clip_path=clip_path,
                    best_frame=best_frame,
                    audio_pcm=audio_pcm,
                    sample_rate=args.sample_rate,
                    metadata={
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "event_type": last_fusion_event_type or "interrupted",
                        "frame_num": str(frame_num),
                    },
                )

        recorder.wait()
        camera.release()
        if audio_capture:
            audio_capture.stop()
        if pusher:
            pusher.stop()
        if args.show:
            cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  V2 MONITOR — SESSION COMPLETE")
    print("=" * 60)
    print(f"  Total frames:    {frame_num}")
    print(f"  Events recorded: {events_saved}")
    print(f"  Events skipped:  {events_skipped}")
    stats = storage.get_stats()
    print(f"  Clips on disk:   {stats['clip_count']} "
          f"({stats['total_size_mb']}MB / {stats['budget_gb']}GB)")
    print(f"  Oldest clip:     {stats['oldest_hours']}h ago")
    print(f"  Free disk:       {stats['free_disk_gb']}GB")
    print(f"  Mode:            {'video+audio' if audio_capture else 'video-only'}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Activity Monitor V2 — Edge Pipeline"
    )

    # Video source
    parser.add_argument("--live", action="store_true",
                        help="Use live camera (default: video file)")
    parser.add_argument("--input", "-i", default=None,
                        help="Path to test video file")
    parser.add_argument("--device", default=None,
                        help="Camera device (/dev/video2 or 0 for default)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15,
                        help="Camera FPS (default: 15)")
    parser.add_argument("--rotate", type=int, default=0,
                        choices=[0, 90, 180, 270])
    parser.add_argument("--show", action="store_true",
                        help="Show live preview window")

    # Audio
    parser.add_argument("--mic", default=None,
                        help="Mic device (plughw:3,0 or None for default)")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio (video-only mode)")

    # Motion detection
    parser.add_argument("--min-area", type=int, default=3000)
    parser.add_argument("--min-motion-pct", type=float, default=0.5)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--min-solidity", type=float, default=0.3)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--match-distance", type=int, default=60)
    parser.add_argument("--capture-interval", type=float, default=0.066,
                        help="Seconds between frame grabs (default: 0.066 = ~15fps)")

    # YOLO
    parser.add_argument("--yolo-confidence", type=float, default=0.35)

    # Dedup
    parser.add_argument("--phash-threshold", type=int, default=10)

    # Dynamic clip recording
    parser.add_argument("--pre-roll", type=float, default=10.0,
                        help="Ring buffer pre-roll seconds (default: 10)")
    parser.add_argument("--post-roll", type=float, default=5.0,
                        help="Post-roll seconds after trigger ends (default: 5)")
    parser.add_argument("--max-clip-seconds", type=float, default=300,
                        help="Max clip length before forced split (default: 300 = 5min)")
    parser.add_argument("--clip-dir", default="data/clips")

    # Rolling storage
    parser.add_argument("--max-storage-gb", type=float, default=3.0,
                        help="Max total clip storage in GB (default: 3.0)")
    parser.add_argument("--min-free-gb", type=float, default=2.0,
                        help="Minimum free disk space in GB (default: 2.0)")

    # Server
    parser.add_argument("--vision-server", default=None,
                        help="GPU server URL (e.g. http://10.0.0.181:8000)")
    parser.add_argument("--vision-api-key",
                        default="wildlife-vision-secret-2026")
    parser.add_argument("--camera-id", default="pi-outdoor")

    # Heartbeat
    parser.add_argument("--heartbeat-minutes", type=int, default=30)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
