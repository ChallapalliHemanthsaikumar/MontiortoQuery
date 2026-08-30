"""Experiment configuration and recording."""

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    experiment_id: str
    date: str
    food_type: str = "sugar_water"
    food_quantity: str = "5ml"
    observation_duration_sec: int = 300
    camera_resolution: str = "640x480"
    camera_fps: int = 15
    environmental_notes: str = ""
    cv_version: str = "0.1.0"

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def create_experiment(experiment_id=None, **kwargs):
    if experiment_id is None:
        experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    config = ExperimentConfig(
        experiment_id=experiment_id,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        **kwargs,
    )

    output_dir = os.path.join("experiments", experiment_id)
    config.save(output_dir)
    print(f"Experiment created: {experiment_id}")
    print(f"Config saved to: {output_dir}/config.json")
    return config, output_dir
