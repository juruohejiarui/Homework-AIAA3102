"""Strict, read-only final verification of the IEEE report and frozen artifacts.

Run this with the bundled document Python runtime because it uses pypdf. The
script does not fit a model, rerun an experiment, or modify any source labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "main.tex"
PDF = ROOT / "report.pdf"
OUTPUT = ROOT / "report" / "final_verification.json"
CHECKS: list[dict[str, str]] = []


def check(name: str, condition: bool, detail: str) -> None:
    CHECKS.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["y_true"].astype(int).to_numpy()
    p = frame["y_pred"].astype(int).to_numpy()
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision_target_1": precision, "recall_target_1": recall,
        "f1_target_1": f1, "accuracy": (tn + tp) / len(frame),
        "true_negative": tn, "false_positive": fp,
        "false_negative": fn, "true_positive": tp,
    }


def close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (float, int, np.floating, np.integer)) and isinstance(right, (float, int, np.floating, np.integer)):
        return bool(np.isclose(float(left), float(right), rtol=0, atol=tolerance))
    return str(left) == str(right)


def main() -> None:
    source = REPORT.read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "configs/frozen_decisions.json").read_text(encoding="utf-8"))
    asset_validation = json.loads((ROOT / "report_assets/validation_report.json").read_text(encoding="utf-8"))
    source_validation = json.loads((ROOT / "report/source_validation.json").read_text(encoding="utf-8"))

    check("standard IEEE conference class", r"\documentclass[conference]{IEEEtran}" in source, "IEEEtran conference mode")
    check("verified authors", "LAI Jiaxing" in source and "HE Jiarui" in source, "both supplied names")
    check("no placeholders", not re.search(r"\[(?:author|department|city|verified)|\b(?:TODO|TBD|PLACEHOLDER)\b", source, re.I), "source scanned")
    required_sections = [
        "Project Problem and Goal", "Methodology", "Main Evidence and Results", "Case Analysis",
        "Difficulties and Solutions", "AI Usage Declaration", "Discussion and Limitations", "Conclusion",
    ]
    check("required section completeness", all(f"\\section{{{name}}}" in source for name in required_sections), ", ".join(required_sections))
    check("all five ticket discussions", all(f"Ticket {ticket}:" in source for ticket in range(1, 6)), "Tickets 1-5 present")
    check("held-out policy explicit", "held-out tuning" in source.lower() and "held-out was not used for selection" in source.lower(), "dev selection and held-out reporting distinguished")

    check("asset generator validation", asset_validation["status"] == "PASS" and len(asset_validation["checks"]) == 196 and all(x["status"] == "PASS" for x in asset_validation["checks"]), "196/196 checks")
    check("source validation", source_validation["status"] == "PASS" and len(source_validation["checks"]) == 123 and all(x["status"] == "PASS" for x in source_validation["checks"]), "123/123 checks")
    check("held-out labels unchanged", asset_validation["heldout_labels_modified"] is False and asset_validation["heldout_rows_removed"] == 0, "no mutation/removal")

    check("frozen manifest ticket count", [d["ticket"] for d in manifest["decisions"]] == [1, 2, 3, 4, 5], "five ordered decisions")
    check("data hash", sha256(ROOT / manifest["data_path"]) == manifest["data_sha256"], manifest["data_path"])
    check("split hash", sha256(ROOT / manifest["split_path"]) == manifest["split_sha256"], manifest["split_path"])
    check("requirements hash", sha256(ROOT / "requirements-lock.txt") == manifest["requirements_lock_sha256"], "requirements-lock.txt")
    for decision in manifest["decisions"]:
        ticket = decision["ticket"]
        check(f"ticket {ticket} dev prediction hash", sha256(ROOT / decision["archived_dev_predictions"]) == decision["archived_dev_predictions_sha256"], decision["archived_dev_predictions"])
        check(f"ticket {ticket} held-out prediction hash", sha256(ROOT / decision["archived_heldout_predictions"]) == decision["archived_heldout_predictions_sha256"], decision["archived_heldout_predictions"])
        check(f"ticket {ticket} frozen configuration hash", sha256(ROOT / decision["freeze_path"]) == decision["freeze_sha256"], decision["freeze_path"])
        check(f"ticket {ticket} held-out completion hash", sha256(ROOT / decision["heldout_completion_path"]) == decision["heldout_completion_sha256"], decision["heldout_completion_path"])
        check(f"ticket {ticket} dev-only decision", decision["heldout_used_for_selection"] is False and decision["selection_reopening_permitted"] is False, decision["decision_basis_split"])

    dev_table = pd.read_csv(ROOT / "report_assets/tables/table_04_dev_metric_comparison.csv").set_index("ticket")
    held_table = pd.read_csv(ROOT / "report_assets/tables/table_05_heldout_metric_comparison.csv").set_index("ticket")
    confusion = pd.read_csv(ROOT / "report_assets/tables/table_06_confusion_matrix_summary.csv")
    summary = pd.read_csv(ROOT / "report_assets/tables/table_03_five_ticket_summary.csv").set_index("ticket")
    for decision in manifest["decisions"]:
        ticket = decision["ticket"]
        for split_name, table, prediction_key in (
            ("dev", dev_table, "archived_dev_predictions"),
            ("heldout", held_table, "archived_heldout_predictions"),
        ):
            observed = metrics(pd.read_csv(ROOT / decision[prediction_key]))
            expected = decision[f"expected_{split_name}_metrics"]
            for key, value in observed.items():
                check(f"ticket {ticket} {split_name} {key}", close(value, expected[key]) and close(value, table.loc[ticket, key]), f"recomputed={value}")
            row = confusion[(confusion["ticket"] == ticket) & (confusion["split"] == split_name)].iloc[0]
            for short, key in (("tn", "true_negative"), ("fp", "false_positive"), ("fn", "false_negative"), ("tp", "true_positive")):
                check(f"ticket {ticket} {split_name} confusion {short}", int(row[short]) == int(observed[key]), str(int(row[short])))
        check(f"ticket {ticket} summary metrics", close(summary.loc[ticket, "dev_f1"], dev_table.loc[ticket, "f1_target_1"]) and close(summary.loc[ticket, "heldout_f1"], held_table.loc[ticket, "f1_target_1"]), "summary matches prediction recomputation")

    cases = pd.read_csv(ROOT / "report_assets/tables/table_12_representative_case_analysis.csv")
    data = pd.read_csv(ROOT / "data/train.csv").set_index("id")
    audit = pd.read_csv(ROOT / "results/data_quality_audit.csv").set_index("id")
    final_predictions = pd.read_csv(ROOT / "predictions/final-heldout-predictions.csv").set_index("id")
    check("representative case count and IDs", len(cases) == 16 and cases["id"].is_unique, f"rows={len(cases)}")
    for case in cases.itertuples(index=False):
        stable_id = int(case.id)
        check(f"case {stable_id} source truth", int(data.loc[stable_id, "target"]) == int(case.true_label) and str(data.loc[stable_id, "text"]) == str(case.text), "stable ID text and label")
        sources = str(case.source_file).split("; ")
        if len(sources) == 1:
            row = pd.read_csv(ROOT / sources[0]).query("id == @stable_id").iloc[0]
            prefix = "probe" if "probe_y_pred" in row.index else "candidate"
            checks = [
                close(row["baseline_y_pred"], case.baseline_pred), close(row["baseline_score"], case.baseline_score),
                close(row[f"{prefix}_y_pred"], case.candidate_pred), close(row[f"{prefix}_score"], case.candidate_score),
                close(row["outcome"], case.transition_or_disposition), close(row["y_true"], case.true_label), close(row["text"], case.text),
            ]
        else:
            checks = [
                stable_id in audit.index, stable_id in final_predictions.index,
                close(audit.loc[stable_id, "disposition"], case.transition_or_disposition),
                close(final_predictions.loc[stable_id, "y_pred"], case.candidate_pred),
                close(final_predictions.loc[stable_id, "score"], case.candidate_score),
                close(final_predictions.loc[stable_id, "y_true"], case.true_label),
            ]
        check(f"case {stable_id} predictions/scores/category", all(checks), sources[0])

    sweep_source = pd.read_csv(ROOT / "results/threshold_sweep.csv")[["ticket", "threshold", "precision_target_1", "recall_target_1", "f1_target_1"]]
    sweep_figure = pd.read_csv(ROOT / "report_assets/figures/figure_01_threshold_sweep_data.csv")
    check("threshold figure source", sweep_source["ticket"].equals(sweep_figure["ticket"]) and np.allclose(sweep_source.drop(columns="ticket"), sweep_figure.drop(columns="ticket"), rtol=0, atol=1e-15), "61 rows match")
    f1_figure = pd.read_csv(ROOT / "report_assets/figures/figure_02_dev_heldout_f1_data.csv")
    check("dev/held-out figure source", len(f1_figure) == 10 and all(close(row.f1_target_1, (dev_table if row.split == "dev" else held_table).loc[row.ticket, "f1_target_1"]) for row in f1_figure.itertuples()), "10 ticket/split values")
    transition_table = pd.read_csv(ROOT / "report_assets/tables/table_07_prediction_transitions.csv")
    transition_figure = pd.read_csv(ROOT / "report_assets/figures/figure_03_prediction_transitions_data.csv")
    expected_transition_figure = transition_table[transition_table["split"] == "heldout"].reset_index(drop=True)
    check("transition figure source", transition_figure.equals(expected_transition_figure), "five held-out rows")
    disposition_figure = pd.read_csv(ROOT / "report_assets/figures/figure_04_data_quality_dispositions_data.csv").set_index("disposition")["audited_rows"].sort_index()
    disposition_expected = audit.reset_index().groupby("disposition").size().sort_index()
    check("data-quality figure source", disposition_figure.equals(disposition_expected), disposition_figure.to_dict().__str__())

    # Every long decimal in the hand-written source must agree, at its printed
    # precision, with a number found in a verified CSV/JSON evidence artifact.
    known: set[str] = set()
    evidence_roots = [ROOT / name for name in ("results", "predictions", "experiments", "configs", "starter")]
    number_pattern = re.compile(r"(?<![\w.])-?\d+\.\d+(?:[eE][+-]?\d+)?")
    for evidence_root in evidence_roots:
        for path in evidence_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
                try:
                    evidence_text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for token in number_pattern.findall(evidence_text):
                    value = float(token)
                    for digits in range(6, 16):
                        known.add(f"{value:.{digits}f}")
                        known.add(f"{abs(value):.{digits}f}")
    derived_values = [
        abs(float(held_table.loc[4, key]) - float(held_table.loc[1, key]))
        for key in ("precision_target_1", "recall_target_1", "f1_target_1", "accuracy")
    ]
    correction = pd.read_csv(ROOT / "experiments/ticket-5/dev/correction_experiment/dev_metrics.csv")
    derived_values.append(abs(float(correction.iloc[1]["f1_target_1"]) - float(correction.iloc[0]["f1_target_1"])))
    for value in derived_values:
        for digits in range(6, 16):
            known.add(f"{value:.{digits}f}")
    report_decimals = set(re.findall(r"(?<![\d.])-?\d+\.\d{6,}(?!\d)", source))
    unmatched = sorted(token for token in report_decimals if token not in known)
    check("prose decimal provenance", not unmatched, f"checked={len(report_decimals)}, unmatched={unmatched}")

    reader = PdfReader(str(PDF))
    check("PDF opens and page count", len(reader.pages) == 12, f"pages={len(reader.pages)}")
    extracted_pages = [(page.extract_text() or "").strip() for page in reader.pages]
    check("no blank or corrupted PDF pages", all(len(text) >= 300 for text in extracted_pages), str([len(text) for text in extracted_pages]))
    check("US-letter PDF pages", all(abs(float(page.mediabox.width) - 612) < 0.1 and abs(float(page.mediabox.height) - 792) < 0.1 for page in reader.pages), "612x792 pt")
    pdf_text = "\n".join(extracted_pages)
    check("PDF author names", "LAI Jiaxing" in pdf_text and "HE Jiarui" in pdf_text, "front matter")
    check("PDF table coverage", all(f"TABLE {roman}" in pdf_text for roman in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV")), "Tables I-XIV")
    check("PDF figure coverage", all(f"Fig. {number}." in pdf_text for number in range(1, 5)), "Figures 1-4")
    compact_pdf_text = re.sub(r"[^A-Z0-9]", "", pdf_text.upper().replace("\ufb01", "FI").replace("\ufb02", "FL"))
    check("PDF section coverage", all(re.sub(r"[^A-Z0-9]", "", name.upper()) in compact_pdf_text for name in required_sections), "all required sections extracted")
    check("PDF has no placeholder text", not re.search(r"\b(?:TODO|TBD|PLACEHOLDER)\b|\[Author", pdf_text, re.I), "extracted text scanned")

    result = {
        "status": "PASS",
        "scope": "final report, generated assets, frozen artifacts, representative cases, and PDF structure",
        "checks": CHECKS,
        "check_count": len(CHECKS),
        "representative_case_ids": cases["id"].astype(int).tolist(),
        "pdf_pages": len(reader.pages),
        "pdf_sha256": sha256(PDF),
        "experiments_rerun": False,
        "heldout_labels_modified": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(CHECKS), "cases": len(cases), "pages": len(reader.pages), "pdf_sha256": result["pdf_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
