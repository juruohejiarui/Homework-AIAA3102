#!/usr/bin/env python3
"""Run train, evaluate, and predict for every config file."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

CONFIGS_DIR = Path("configs")


def run(cmd: list[str]) -> bool:
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    configs = sorted(CONFIGS_DIR.glob("*.json"))
    if not configs:
        raise FileNotFoundError("no config files found under configs/")

    print(f"Found {len(configs)} configs")

    for config_path in configs:
        config = json.loads(config_path.read_text())
        run_id = config["run_id"]
        checkpoint = Path(config["checkpoint_dir"]) / f"{run_id}_best.pt"

        print(f"\n{'=' * 60}\n  {run_id}\n{'=' * 60}")

        if not run([
            "uv", "run", "python", "-m", "plant_pathology.train",
            "--config", str(config_path),
        ]):
            print(f"  TRAIN FAILED, skipping remaining steps for {run_id}")
            continue

        submissions_dir = Path("predictions")
        submissions_dir.mkdir(parents=True, exist_ok=True)
        submission_out = submissions_dir / f"submission_{run_id}.csv"

        run([
            "uv", "run", "python", "-m", "plant_pathology.evaluate",
            "--config", str(config_path), "--checkpoint", str(checkpoint),
        ])
        run([
            "uv", "run", "python", "-m", "plant_pathology.predict",
            "--config", str(config_path), "--checkpoint", str(checkpoint),
            "--output", str(submission_out),
        ])

    print("\nAll configs processed.")


if __name__ == "__main__":
    main()
