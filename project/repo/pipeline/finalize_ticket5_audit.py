"""Validate curated Ticket 5 records and write the required audit table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .artifacts import DATA_QUALITY_AUDIT_COLUMNS, DATA_QUALITY_DISPOSITIONS, write_csv_artifact, write_json_artifact, write_text_artifact
from .data import load_labeled_tweets
from .data_quality import validate_data_quality_audit
from .run_ticket5_dev import sha256
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_RECORDS_PATH = PROJECT_ROOT / "experiments" / "ticket-5" / "final_audit_records.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "data_quality_audit.csv"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise RuntimeError("data-quality audit already exists; refusing overwrite")
    completion_path = PROJECT_ROOT / "experiments" / "ticket-5" / "heldout" / "heldout_evaluation_completed.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["ticket5_heldout_reporting_count"] != 1 or completion["selection_reopened"] is not False:
        raise RuntimeError("Ticket 5 post-freeze held-out chronology is invalid")
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    if payload.get("created_after_ticket5_freeze") is not True or payload.get("used_to_reopen_model_selection") is not False:
        raise ValueError("final audit chronology declaration is invalid")
    frame = pd.DataFrame(payload["records"], columns=DATA_QUALITY_AUDIT_COLUMNS)
    if frame["id"].duplicated().any():
        raise ValueError("final audit uses one disposition per stable ID")
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    validate_data_quality_audit(frame, valid_ids=set(data["id"].astype(int)))
    if set(frame["disposition"]) != DATA_QUALITY_DISPOSITIONS:
        raise ValueError("final audit must demonstrate all required dispositions")
    write_csv_artifact(frame, args.output)
    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.finalize_ticket5_audit", *sys.argv[1:]])
    write_json_artifact(
        {
            "scope": "Post-freeze curation of evidence-based Ticket 5 audit records",
            "exact_command": command,
            "records_sha256": sha256(args.records),
            "output_sha256": sha256(args.output),
            "data_sha256": sha256(args.data),
            "split_sha256": sha256(args.split),
            "rows": len(frame),
            "unique_ids": int(frame["id"].nunique()),
            "disposition_counts": {key: int(value) for key, value in frame["disposition"].value_counts().sort_index().items()},
            "source_dataset_modified": False,
            "heldout_labels_modified": False,
            "heldout_rows_removed": 0,
            "selection_reopened": False,
        },
        PROJECT_ROOT / "experiments" / "ticket-5" / "final_audit_manifest.json",
    )
    write_text_artifact(command, PROJECT_ROOT / "experiments" / "ticket-5" / "final_audit_command.txt")
    print(frame["disposition"].value_counts().sort_index().to_string())
    print(f"audit_rows={len(frame)} unique_ids={frame['id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
