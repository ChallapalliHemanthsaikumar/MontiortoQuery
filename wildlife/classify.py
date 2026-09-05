"""Classify wildlife in captured frames using YOLOv8-nano.

Runs on the laptop GPU/CPU — not on the Pi.
Downloads the model on first run (~6MB for yolov8n).
"""

import os
import json
import csv
import cv2
from datetime import datetime, timezone

WILDLIFE_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "squirrel", "rabbit", "deer", "raccoon", "fox",
    "mouse", "rat", "chipmunk",
}

COCO_OUTDOOR_ANIMALS = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}


class WildlifeClassifier:
    """YOLOv8-nano classifier for wildlife species detection."""

    def __init__(self, model_name="yolov8n.pt", confidence=0.35, device=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run:\n"
                "  pip install ultralytics\n"
                "This downloads YOLOv8-nano (~6MB) on first use."
            )

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device

    def classify(self, image_path):
        results = self.model(
            image_path,
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                is_wildlife = (cls_name in WILDLIFE_CLASSES or
                               cls_id in COCO_OUTDOOR_ANIMALS)

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "is_wildlife": is_wildlife,
                })

        return detections

    def classify_and_annotate(self, image_path, output_path=None):
        detections = self.classify(image_path)
        frame = cv2.imread(image_path)
        if frame is None:
            return detections, None

        wildlife_found = [d for d in detections if d["is_wildlife"]]
        for det in wildlife_found:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)

        return detections, frame


def batch_classify(image_dir, output_dir=None, confidence=0.35):
    """Classify all images in a directory."""
    classifier = WildlifeClassifier(confidence=confidence)

    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(root, f))

    print(f"Classifying {len(images)} images...")
    results = []

    for i, img_path in enumerate(images):
        detections = classifier.classify(img_path)
        wildlife = [d for d in detections if d["is_wildlife"]]

        if output_dir and wildlife:
            rel = os.path.relpath(img_path, image_dir)
            out_path = os.path.join(output_dir, "annotated", rel)
            classifier.classify_and_annotate(img_path, out_path)

        results.append({
            "image": img_path,
            "detections": detections,
            "wildlife": wildlife,
            "wildlife_count": len(wildlife),
            "species": list(set(d["class_name"] for d in wildlife)),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            wildlife_total = sum(r["wildlife_count"] for r in results)
            print(f"  [{i+1}/{len(images)}] "
                  f"{wildlife_total} wildlife detections so far")

    return results


def save_classification_report(results, output_dir):
    """Save classification results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "classifications.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(output_dir, "classifications.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "species", "confidence", "bbox", "wildlife_count"])
        for r in results:
            if r["wildlife"]:
                for w in r["wildlife"]:
                    writer.writerow([
                        r["image"], w["class_name"], w["confidence"],
                        json.dumps(w["bbox"]), r["wildlife_count"]
                    ])
            else:
                writer.writerow([r["image"], "", "", "", 0])

    species_counts = {}
    for r in results:
        for s in r["species"]:
            species_counts[s] = species_counts.get(s, 0) + 1

    summary = {
        "total_images": len(results),
        "images_with_wildlife": sum(1 for r in results if r["wildlife_count"] > 0),
        "total_detections": sum(r["wildlife_count"] for r in results),
        "species_counts": species_counts,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = os.path.join(output_dir, "classification_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nClassification Results:")
    print(f"  Total images:        {summary['total_images']}")
    print(f"  With wildlife:       {summary['images_with_wildlife']}")
    print(f"  Total detections:    {summary['total_detections']}")
    print(f"  Species found:       {species_counts}")
    print(f"\n  Reports saved to:    {output_dir}")

    return summary
