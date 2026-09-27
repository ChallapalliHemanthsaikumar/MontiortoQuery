"""Fusion gate — combine visual + audio signals to decide SAVE or SKIP.

The fusion gate is a decision matrix. It takes two inputs:
    1. Visual signal: YOLO detection result (person/animal/vehicle/nothing)
    2. Audio signal: YAMNet classification result (speech/bark/engine/wind/silence)

And outputs one of:
    - SAVE: record clip, push to server, run full 9-step pipeline
    - SKIP: ignore this moment, keep monitoring

The key insight: neither sensor alone is reliable.
    - Camera sees motion in wind → false positive
    - Mic hears distant traffic → not interesting
    - Camera sees person + mic hears speech → DEFINITELY interesting
    - No motion + mic hears glass break → SAVE even without visual

This is sensor fusion. Same principle as self-driving cars:
camera + lidar + radar → confident decision.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Decision(Enum):
    SAVE = "save"
    SKIP = "skip"


class EventType(Enum):
    PERSON_SPEECH = "person_speech"
    PERSON_SILENT = "person_silent"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    AUDIO_ONLY = "audio_only"
    UNCLASSIFIED_MOTION = "unclassified_motion"
    HEARTBEAT = "heartbeat"


@dataclass
class FusionResult:
    decision: Decision
    event_type: Optional[EventType]
    confidence: float
    visual_class: str
    audio_class: str
    audio_priority: str
    reason: str


VISUAL_PERSON_CLASSES = {"person"}
VISUAL_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
}
VISUAL_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class FusionGate:
    """Decision matrix combining YOLO visual + YAMNet audio signals.

    The gate runs after BOTH sensors have produced results for the current
    moment. It doesn't do any ML itself — it's pure logic rules based on
    the combination of signals.
    """

    def __init__(self, require_visual_for_medium_audio=True):
        self.require_visual_for_medium_audio = require_visual_for_medium_audio

    def decide(self, motion_triggered, yolo_detections, audio_results,
               audio_priority):
        """Make a SAVE/SKIP decision based on combined signals.

        Args:
            motion_triggered: bool from motion detector
            yolo_detections: list of dicts from YOLO classifier
                             [{"class_name": "person", "confidence": 0.91}, ...]
            audio_results: list of (class_name, confidence) from YAMNet
            audio_priority: "high", "medium", "low", "ignore" from classifier

        Returns:
            FusionResult with decision and metadata
        """
        visual_class = self._get_visual_class(yolo_detections)
        audio_class, audio_conf = self._get_audio_class(audio_results)
        visual_conf = self._get_visual_confidence(yolo_detections)

        has_person = visual_class == "person"
        has_animal = visual_class == "animal"
        has_vehicle = visual_class == "vehicle"
        has_visual = has_person or has_animal or has_vehicle
        is_speech = audio_class in ("Speech", "Conversation", "Narration, monologue")
        is_high_audio = audio_priority == "high"
        is_medium_audio = audio_priority == "medium"

        combined_conf = max(visual_conf, audio_conf)

        # Rule 1: Person + speech = highest priority
        if has_person and is_speech:
            return FusionResult(
                Decision.SAVE, EventType.PERSON_SPEECH, combined_conf,
                visual_class, audio_class, audio_priority,
                "person detected with speech audio"
            )

        # Rule 2: Person + any audio (including silence)
        if has_person:
            return FusionResult(
                Decision.SAVE, EventType.PERSON_SILENT, visual_conf,
                visual_class, audio_class, audio_priority,
                "person detected"
            )

        # Rule 3: Animal detected
        if has_animal:
            return FusionResult(
                Decision.SAVE, EventType.ANIMAL, combined_conf,
                visual_class, audio_class, audio_priority,
                "animal detected"
            )

        # Rule 4: Vehicle detected
        if has_vehicle:
            return FusionResult(
                Decision.SAVE, EventType.VEHICLE, combined_conf,
                visual_class, audio_class, audio_priority,
                "vehicle detected"
            )

        # Rule 5: No visual detection but high-priority audio
        # (speech off-camera, glass break, siren, dog bark)
        if is_high_audio:
            return FusionResult(
                Decision.SAVE, EventType.AUDIO_ONLY, audio_conf,
                visual_class, audio_class, audio_priority,
                f"high-priority audio: {audio_class} (no visual match)"
            )

        # Rule 6: Motion + medium-priority audio
        # (motion + engine sound = car not yet in frame)
        if motion_triggered and is_medium_audio:
            if not self.require_visual_for_medium_audio:
                return FusionResult(
                    Decision.SAVE, EventType.UNCLASSIFIED_MOTION, audio_conf,
                    visual_class, audio_class, audio_priority,
                    f"motion with {audio_class} audio"
                )

        # Rule 7: Motion + low/ignore audio = probably wind/leaves
        if motion_triggered and audio_priority in ("ignore", "low"):
            return FusionResult(
                Decision.SKIP, None, 0.0,
                visual_class, audio_class, audio_priority,
                f"motion rejected: audio is {audio_class} ({audio_priority})"
            )

        # Rule 8: Motion + no YOLO match + no notable audio
        if motion_triggered and not has_visual:
            return FusionResult(
                Decision.SKIP, None, 0.0,
                visual_class, audio_class, audio_priority,
                "motion with no visual or audio confirmation"
            )

        # Default: nothing happening
        return FusionResult(
            Decision.SKIP, None, 0.0,
            visual_class, audio_class, audio_priority,
            "no significant activity"
        )

    def _get_visual_class(self, detections):
        if not detections:
            return "nothing"
        for d in detections:
            if d["class_name"] in VISUAL_PERSON_CLASSES:
                return "person"
        for d in detections:
            if d["class_name"] in VISUAL_ANIMAL_CLASSES:
                return "animal"
        for d in detections:
            if d["class_name"] in VISUAL_VEHICLE_CLASSES:
                return "vehicle"
        return "object"

    def _get_visual_confidence(self, detections):
        if not detections:
            return 0.0
        return max(d["confidence"] for d in detections)

    def _get_audio_class(self, results):
        if not results:
            return "silence", 0.0
        return results[0][0], results[0][1]
