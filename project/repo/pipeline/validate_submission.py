"""Validate final documentation and machine-readable submission artifacts without refitting models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .artifacts import (
    DATA_QUALITY_AUDIT_COLUMNS,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    THRESHOLD_SWEEP_COLUMNS,
    validate_prediction_frame,
)
from .data_quality import validate_data_quality_audit
from .splits import load_fixed_split


ROOT = Path(__file__).resolve().parents[1]


def _require_text(path: Path, phrases: list[str]) -> None:
    text = path.read_text(encoding="utf-8").lower()
    missing = [phrase for phrase in phrases if phrase.lower() not in text]
    if missing:
        raise AssertionError(f"{path.relative_to(ROOT)} lacks required text: {missing}")


def main() -> int:
    required_paths = [
        "pipeline",
        "tickets",
        "experiments",
        "predictions",
        "results",
        "logs/chat.md",
        "report/main.tex",
        "report/references.bib",
        "report_assets",
        "report.pdf",
        "README.md",
    ]
    missing_paths = [path for path in required_paths if not (ROOT / path).exists()]
    if missing_paths:
        raise AssertionError(f"required submission paths are missing: {missing_paths}")

    split = load_fixed_split(ROOT / "starter" / "data" / "split_indices.json")
    final_predictions = pd.read_csv(ROOT / "predictions" / "final-heldout-predictions.csv")
    if list(final_predictions.columns) != PREDICTION_COLUMNS:
        raise AssertionError("final prediction schema is incorrect")
    validate_prediction_frame(final_predictions, expected_ids=list(split.heldout_ids))
    if not final_predictions["ticket"].eq("ticket_5_final_frozen_decision").all():
        raise AssertionError("final prediction provenance is incorrect")

    summary = pd.read_csv(ROOT / "results" / "summary.csv")
    if list(summary.columns) != SUMMARY_COLUMNS or summary["ticket"].tolist() != [f"ticket_{i}" for i in range(1, 6)]:
        raise AssertionError("summary schema or ticket order is incorrect")
    sweep = pd.read_csv(ROOT / "results" / "threshold_sweep.csv")
    if list(sweep.columns) != THRESHOLD_SWEEP_COLUMNS or len(sweep) != 61:
        raise AssertionError("threshold sweep schema or row count is incorrect")
    quality = pd.read_csv(ROOT / "results" / "data_quality_audit.csv")
    if list(quality.columns) != DATA_QUALITY_AUDIT_COLUMNS:
        raise AssertionError("data-quality schema is incorrect")
    validate_data_quality_audit(quality, valid_ids=set(split.train_ids + split.dev_ids + split.heldout_ids))

    verification = json.loads((ROOT / "experiments" / "final-reproducibility-audit" / "reproducibility_verification.json").read_text(encoding="utf-8"))
    if verification["result"] != "PASS" or verification["selection_reopened"]:
        raise AssertionError("frozen-decision reproducibility audit is not a closed PASS")

    evidence_terms = [
        "hypothesis",
        "intended lever",
        "controlled setup",
        "dev evidence",
        "frozen decision",
        "held-out evidence",
        "concrete prediction changes",
        "interpretation",
        "limitation",
    ]
    tickets = [
        "ticket-1-baseline.md",
        "ticket-2-normalization.md",
        "ticket-3-shortcuts.md",
        "ticket-4-decision-rule.md",
        "ticket-5-data-quality.md",
    ]
    for ticket in tickets:
        _require_text(ROOT / "tickets" / ticket, evidence_terms)

    _require_text(
        ROOT / "README.md",
        [
            "environment setup",
            "data acquisition",
            "expected directory structure",
            "reproduce the baseline",
            "reproduce each ticket",
            "regenerate final artifacts",
            "validate the submission",
        ],
    )
    _require_text(ROOT / "logs" / "chat.md", ["ai usage", "verification record", "limitations"])

    report = ROOT / "report.pdf"
    report_bytes = report.read_bytes()
    if len(report_bytes) < 20_000 or not report_bytes.startswith(b"%PDF-"):
        raise AssertionError("report.pdf is not a plausible PDF")
    _require_text(
        ROOT / "report" / "main.tex",
        [
            "abstract",
            "project problem and goal",
            "methodology",
            "main evidence and results",
            "case analysis",
            "difficulties and solutions",
            "ai usage declaration",
            "discussion and limitations",
            "conclusion",
        ],
    )

    result = {
        "result": "PASS",
        "tickets": 5,
        "summary_rows": len(summary),
        "threshold_rows": len(sweep),
        "audit_rows": len(quality),
        "heldout_prediction_rows": len(final_predictions),
        "refit_performed": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
