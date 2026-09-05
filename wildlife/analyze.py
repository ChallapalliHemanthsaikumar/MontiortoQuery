"""One-command analysis: classify all captures + generate timelapse + report.

Run on laptop after copying data from Pi:
    python -m wildlife.analyze --data-dir data/wildlife/wildlife001
"""

import argparse
import os
import json
import csv
from collections import Counter
from datetime import datetime

from wildlife.classify import batch_classify, save_classification_report
from wildlife.timelapse import generate_timelapse


def load_events_csv(data_dir, experiment_id):
    """Load the capture event log from the Pi."""
    csv_path = os.path.join(data_dir, f"{experiment_id}_events.csv")
    if not os.path.exists(csv_path):
        print(f"  No event log found at {csv_path}")
        return []
    events = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def generate_activity_report(events, classifications, output_dir):
    """Generate a combined activity report from Pi events + laptop classification."""
    os.makedirs(output_dir, exist_ok=True)

    hourly = Counter()
    for ev in events:
        try:
            ts = datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00"))
            hourly[ts.strftime("%Y-%m-%d %H:00")] += 1
        except (KeyError, ValueError):
            pass

    triggers = Counter(ev.get("trigger", "unknown") for ev in events)

    species_by_hour = {}
    for c in classifications:
        for sp in c["species"]:
            img_name = os.path.basename(c["image"])
            ts_part = img_name.split("_f")[0]
            try:
                ts = datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
                hour_key = ts.strftime("%Y-%m-%d %H:00")
            except ValueError:
                hour_key = "unknown"
            if hour_key not in species_by_hour:
                species_by_hour[hour_key] = Counter()
            species_by_hour[hour_key][sp] += 1

    all_species = Counter()
    for c in classifications:
        for sp in c["species"]:
            all_species[sp] += 1

    busiest_hours = hourly.most_common(10)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "capture_stats": {
            "total_events": len(events),
            "triggers": dict(triggers),
            "busiest_hours": [{"hour": h, "captures": c} for h, c in busiest_hours],
        },
        "classification_stats": {
            "total_classified": len(classifications),
            "with_wildlife": sum(1 for c in classifications if c["wildlife_count"] > 0),
            "species_counts": dict(all_species),
        },
        "activity_by_hour": {
            h: dict(sp) for h, sp in sorted(species_by_hour.items())
        },
    }

    report_path = os.path.join(output_dir, "activity_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print("  WILDLIFE ACTIVITY REPORT")
    print("=" * 55)
    print(f"  Total captures:    {len(events)}")
    print(f"  Triggers:          {dict(triggers)}")
    print(f"  Images classified: {len(classifications)}")
    print(f"  Wildlife found in: "
          f"{sum(1 for c in classifications if c['wildlife_count'] > 0)} images")
    print(f"  Species:           {dict(all_species)}")
    if busiest_hours:
        print(f"  Busiest hour:      {busiest_hours[0][0]} ({busiest_hours[0][1]} captures)")
    print(f"\n  Full report:       {report_path}")
    print("=" * 55)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze wildlife captures (run on laptop after copying data from Pi)"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to wildlife data (e.g. data/wildlife/wildlife001)")
    parser.add_argument("--experiment", "-e", default=None,
                        help="Experiment ID (auto-detected from dir name if omitted)")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--timelapse-fps", type=int, default=10,
                        help="Timelapse playback FPS (default: 10)")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip YOLO classification")
    parser.add_argument("--skip-timelapse", action="store_true",
                        help="Skip timelapse generation")
    args = parser.parse_args()

    data_dir = args.data_dir
    experiment = args.experiment or os.path.basename(data_dir.rstrip("/\\"))
    output_dir = os.path.join(data_dir, "analysis")

    print(f"Analyzing: {data_dir}")
    print(f"Experiment: {experiment}")
    print(f"Output: {output_dir}\n")

    events = load_events_csv(data_dir, experiment)
    print(f"Loaded {len(events)} capture events from Pi log")

    classifications = []
    if not args.skip_classify:
        print("\n--- YOLO Classification ---")
        classifications = batch_classify(data_dir, output_dir, args.confidence)
        save_classification_report(classifications, output_dir)
    else:
        print("Skipping classification (--skip-classify)")

    if not args.skip_timelapse:
        print("\n--- Timelapse ---")
        timelapse_path = os.path.join(output_dir, "timelapse.mp4")
        generate_timelapse(data_dir, timelapse_path, fps=args.timelapse_fps)
    else:
        print("Skipping timelapse (--skip-timelapse)")

    print("\n--- Activity Report ---")
    generate_activity_report(events, classifications, output_dir)


if __name__ == "__main__":
    main()
