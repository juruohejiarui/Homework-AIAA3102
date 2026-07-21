"""Generate verified IEEE-report tables, figures, and case materials.

This script only reads frozen/generated repository artifacts. It does not fit a
model, change labels, alter predictions, or rerun a held-out evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "topic-a-report-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
METRIC_COLS = [
    "precision_target_1",
    "recall_target_1",
    "f1_target_1",
    "accuracy",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]
REQUIRED_PREDICTION_COLS = {"id", "y_true", "y_pred", "score", "model_name", "ticket"}

ASSETS: list[dict[str, str]] = []
CHECKS: list[dict[str, Any]] = []


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source(*paths: str) -> str:
    for value in paths:
        assert (ROOT / value).is_file(), f"Missing source artifact: {value}"
    return "; ".join(paths)


def record_check(name: str, passed: bool, detail: str) -> None:
    CHECKS.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    assert passed, f"{name}: {detail}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str) -> dict[str, Any]:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: str, required: Iterable[str] = ()) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / path)
    missing = set(required) - set(frame.columns)
    assert not missing, f"{path} missing columns: {sorted(missing)}"
    return frame


def compute_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = frame["y_pred"].astype(int).to_numpy()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tn + tp) / len(frame)
    return {
        "precision_target_1": precision,
        "recall_target_1": recall,
        "f1_target_1": f1,
        "accuracy": accuracy,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def transitions(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, int]:
    left = baseline[["id", "y_true", "y_pred"]].rename(columns={"y_pred": "baseline_y_pred"})
    right = candidate[["id", "y_true", "y_pred"]].rename(columns={"y_true": "candidate_y_true", "y_pred": "candidate_y_pred"})
    merged = left.merge(right, on="id", how="inner", validate="one_to_one")
    assert len(merged) == len(left) == len(right)
    assert merged["y_true"].equals(merged["candidate_y_true"])
    y = merged["y_true"]
    b = merged["baseline_y_pred"]
    c = merged["candidate_y_pred"]
    return {
        "fixed_fp": int(((y == 0) & (b == 1) & (c == 0)).sum()),
        "fixed_fn": int(((y == 1) & (b == 0) & (c == 1)).sum()),
        "new_fp": int(((y == 0) & (b == 0) & (c == 1)).sum()),
        "new_fn": int(((y == 1) & (b == 1) & (c == 0)).sum()),
    }


def tex_escape(value: Any) -> str:
    if pd.isna(value):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text).replace("\n", " ")


def display_value(value: Any) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6f}"
    return str(value)


def write_tex_table(frame: pd.DataFrame, path: Path, title: str, label: str, notes: str,
                    display_columns: list[str] | None = None) -> None:
    shown = frame[display_columns].copy() if display_columns else frame.copy()
    align = "l" + "r" * (len(shown.columns) - 1)
    lines = [
        r"\begin{table*}[t]", r"\centering", f"\\caption{{{tex_escape(title)}}}",
        f"\\label{{{label}}}", r"\resizebox{\textwidth}{!}{%", f"\\begin{{tabular}}{{{align}}}",
        r"\hline", " & ".join(tex_escape(column.replace("_", " ")) for column in shown.columns) + r" \\",
        r"\hline",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append(" & ".join(tex_escape(display_value(value)) for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}", f"\\par\\footnotesize{{{tex_escape(notes)}}}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_wrapped_tex_table(frame: pd.DataFrame, path: Path, title: str, label: str,
                            notes: str, display_columns: list[str], widths: list[float]) -> None:
    """Write a readable full-width text table without scaling type to microscopic size."""
    assert len(display_columns) == len(widths)
    shown = frame[display_columns].copy()
    column_spec = "@{}" + "".join(f"p{{{width:.2f}\\textwidth}}" for width in widths) + "@{}"
    lines = [
        r"\begin{table*}[t]", r"\centering", f"\\caption{{{tex_escape(title)}}}",
        f"\\label{{{label}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}", f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline", " & ".join(rf"\textbf{{{tex_escape(column.replace('_', ' '))}}}" for column in shown.columns) + r" \\",
        r"\hline",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append(" & ".join(tex_escape(display_value(value)) for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", f"\\par\\footnotesize{{{tex_escape(notes)}}}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table(name: str, frame: pd.DataFrame, title: str, label: str, sources: str,
                notes: str, display_columns: list[str] | None = None, expected_rows: int | None = None) -> None:
    if expected_rows is not None:
        record_check(f"{name} row count", len(frame) == expected_rows, f"expected {expected_rows}, observed {len(frame)}")
    csv_path = TABLES / f"{name}.csv"
    tex_path = TABLES / f"{name}.tex"
    frame.to_csv(csv_path, index=False, float_format="%.17g", quoting=csv.QUOTE_MINIMAL)
    write_tex_table(frame, tex_path, title, label, notes, display_columns)
    ASSETS.append({
        "asset": rel(csv_path), "asset_type": "table_source_csv", "title": title,
        "split": str(frame["split"].iloc[0]) if "split" in frame and frame["split"].nunique() == 1 else "mixed/explicit by row",
        "comparison_baseline": "Ticket 1 frozen baseline where applicable", "source_artifacts": sources,
        "notes": notes,
    })
    ASSETS.append({
        "asset": rel(tex_path), "asset_type": "report_ready_latex_table", "title": title,
        "split": str(frame["split"].iloc[0]) if "split" in frame and frame["split"].nunique() == 1 else "mixed/explicit by row",
        "comparison_baseline": "Ticket 1 frozen baseline where applicable", "source_artifacts": sources,
        "notes": notes,
    })


def save_figure(fig: plt.Figure, stem: str, title: str, sources: str, notes: str) -> None:
    for extension in ("png", "svg"):
        path = FIGURES / f"{stem}.{extension}"
        metadata = {"Software": "scripts/generate_report_assets.py"} if extension == "png" else {"Date": None, "Creator": "scripts/generate_report_assets.py"}
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight", metadata=metadata)
        ASSETS.append({
            "asset": rel(path), "asset_type": f"figure_{extension}", "title": title,
            "split": "explicit in figure", "comparison_baseline": "Ticket 1 frozen baseline where applicable",
            "source_artifacts": sources, "notes": notes,
        })
    plt.close(fig)


def shorten(text: str, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "verified-report-assets-v1",
    })

    manifest_path = "configs/frozen_decisions.json"
    manifest = load_json(manifest_path)
    decisions = {row["ticket"]: row for row in manifest["decisions"]}
    record_check("five frozen ticket decisions", set(decisions) == {1, 2, 3, 4, 5}, f"tickets={sorted(decisions)}")

    data_path = ROOT / manifest["data_path"]
    split_path = ROOT / manifest["split_path"]
    data = pd.read_csv(data_path)
    split = load_json(manifest["split_path"])
    record_check("source data SHA-256", sha256(data_path) == manifest["data_sha256"], rel(data_path))
    record_check("split SHA-256", sha256(split_path) == manifest["split_sha256"], rel(split_path))
    record_check("source IDs unique", data["id"].is_unique, f"rows={len(data)}")
    split_sets = {name: set(map(int, split[f"{name}_ids"])) for name in ("train", "dev", "heldout")}
    record_check("fixed split disjoint", not (split_sets["train"] & split_sets["dev"] or split_sets["train"] & split_sets["heldout"] or split_sets["dev"] & split_sets["heldout"]), "pairwise intersections are empty")
    record_check("fixed split complete", set(data["id"].astype(int)) == set.union(*split_sets.values()), f"rows={len(data)}")

    predictions: dict[tuple[int, str], pd.DataFrame] = {}
    for ticket, decision in decisions.items():
        for split_name, key in (("dev", "archived_dev_predictions"), ("heldout", "archived_heldout_predictions")):
            path = decision[key]
            frame = load_csv(path, REQUIRED_PREDICTION_COLS)
            expected_hash = decision[f"{key}_sha256"]
            record_check(f"ticket {ticket} {split_name} archived prediction SHA-256", sha256(ROOT / path) == expected_hash, path)
            record_check(f"ticket {ticket} {split_name} row count", len(frame) == len(split_sets[split_name]), f"path={path}, rows={len(frame)}")
            record_check(f"ticket {ticket} {split_name} stable IDs", set(frame["id"].astype(int)) == split_sets[split_name], path)
            record_check(f"ticket {ticket} {split_name} frozen model", set(frame["model_name"].astype(str)) == {decision["model_name"]}, f"expected={decision['model_name']}, observed={sorted(frame['model_name'].astype(str).unique())}")
            truth = data.set_index("id").loc[frame["id"].astype(int), "target"].astype(int).to_numpy()
            record_check(f"ticket {ticket} {split_name} labels unchanged", np.array_equal(truth, frame["y_true"].astype(int).to_numpy()), path)
            calculated = compute_metrics(frame)
            expected = decision[f"expected_{split_name}_metrics"]
            for metric, value in calculated.items():
                tolerance = 0 if isinstance(value, int) else 1e-12
                record_check(f"ticket {ticket} {split_name} {metric}", abs(float(value) - float(expected[metric])) <= tolerance, f"calculated={value}, manifest={expected[metric]}")
            predictions[(ticket, split_name)] = frame

    final_audit = load_json("experiments/ticket-5/final_audit_manifest.json")
    record_check("held-out labels not modified", final_audit["heldout_labels_modified"] is False, "experiments/ticket-5/final_audit_manifest.json")
    record_check("held-out rows not removed", final_audit["heldout_rows_removed"] == 0, "experiments/ticket-5/final_audit_manifest.json")

    # Table 1: dataset and fixed split statistics.
    dataset_rows = []
    for split_name in ("train", "dev", "heldout"):
        subset = data[data["id"].isin(split_sets[split_name])]
        dataset_rows.append({
            "split": split_name, "rows": len(subset), "target_0": int((subset["target"] == 0).sum()),
            "target_1": int((subset["target"] == 1).sum()), "target_1_rate": float(subset["target"].mean()),
            "unique_ids": int(subset["id"].nunique()), "seed": split["seed"],
        })
    dataset_table = pd.DataFrame(dataset_rows)
    write_table("table_01_dataset_split_statistics", dataset_table, "Dataset and Fixed Split Statistics",
                "tab:dataset_split", source(manifest["data_path"], manifest["split_path"]),
                "Counts are tweets; target-1 rate is a fraction in [0,1]. The split is fixed and stratified.", expected_rows=3)

    # Table 2: baseline and assignment-reference comparison.
    floor = load_csv("experiments/step-4-baselines/results/dev_metrics.csv")
    contract = load_json("starter/configs/project_contract.json")
    comparison = load_json("experiments/ticket-1/heldout/primary_contract_comparison.json")
    baseline_rows = []
    for row in floor.to_dict("records"):
        baseline_rows.append({
            "condition": row["model_name"], "split": row["split"], "metric": "f1_target_1",
            "value": row["f1_target_1"], "reference_or_tolerance": np.nan, "absolute_gap": np.nan,
            "within_tolerance": np.nan,
        })
    baseline_rows.append({
        "condition": "frozen_baseline_heldout", "split": "heldout", "metric": comparison["metric"],
        "value": comparison["actual"], "reference_or_tolerance": comparison["reference"],
        "absolute_gap": comparison["absolute_difference"], "within_tolerance": comparison["matches_reference"],
    })
    baseline_reference = pd.DataFrame(baseline_rows)
    write_table("table_02_baseline_reference_comparison", baseline_reference,
                "Baseline Floor and Assignment-Reference Comparison", "tab:baseline_reference",
                source("experiments/step-4-baselines/results/dev_metrics.csv", "starter/configs/project_contract.json", "experiments/ticket-1/heldout/primary_contract_comparison.json"),
                f"F1 is unitless in [0,1]. The assignment reference tolerance is {comparison['tolerance']:.3f}; the frozen held-out result does not match it.", expected_rows=3)

    # Tables 3--7: project summary, split metrics, confusion matrices, transitions.
    summary_rows, metric_rows, confusion_rows, transition_rows = [], [], [], []
    t1_dev, t1_heldout = predictions[(1, "dev")], predictions[(1, "heldout")]
    for ticket, decision in decisions.items():
        dev_metrics = compute_metrics(predictions[(ticket, "dev")])
        heldout_metrics = compute_metrics(predictions[(ticket, "heldout")])
        heldout_transitions = transitions(t1_heldout, predictions[(ticket, "heldout")])
        dev_transitions = transitions(t1_dev, predictions[(ticket, "dev")])
        record_check(f"ticket {ticket} heldout transitions", heldout_transitions == decision["expected_transitions_vs_ticket1_baseline"], str(heldout_transitions))
        summary_rows.append({
            "ticket": ticket, "model_name": decision["model_name"], "decision": decision["decision"],
            "dev_f1": dev_metrics["f1_target_1"], "heldout_f1": heldout_metrics["f1_target_1"],
            "heldout_accuracy": heldout_metrics["accuracy"], **heldout_transitions,
        })
        for split_name, metrics in (("dev", dev_metrics), ("heldout", heldout_metrics)):
            metric_rows.append({"ticket": ticket, "split": split_name, "model_name": decision["model_name"], **metrics})
            confusion_rows.append({
                "ticket": ticket, "split": split_name, "model_name": decision["model_name"],
                "tn": metrics["true_negative"], "fp": metrics["false_positive"],
                "fn": metrics["false_negative"], "tp": metrics["true_positive"],
            })
        transition_rows.extend([
            {"ticket": ticket, "split": "dev", "baseline": "ticket_1_frozen", **dev_transitions},
            {"ticket": ticket, "split": "heldout", "baseline": "ticket_1_frozen", **heldout_transitions},
        ])
    summary_table = pd.DataFrame(summary_rows)
    all_metric_table = pd.DataFrame(metric_rows)
    dev_table = all_metric_table[all_metric_table["split"] == "dev"].reset_index(drop=True)
    heldout_table = all_metric_table[all_metric_table["split"] == "heldout"].reset_index(drop=True)
    confusion_table = pd.DataFrame(confusion_rows)
    transition_table = pd.DataFrame(transition_rows)
    pred_sources = source(manifest_path, "results/summary.csv", *(d["archived_dev_predictions"] for d in decisions.values()), *(d["archived_heldout_predictions"] for d in decisions.values()))
    write_table("table_03_five_ticket_summary", summary_table, "Project-Wide Results Across Five Tickets", "tab:five_ticket_summary", pred_sources,
                "Metrics are recomputed from archived predictions. Transition counts are held-out changes relative to Ticket 1; counts are tweets.",
                display_columns=["ticket", "model_name", "dev_f1", "heldout_f1", "heldout_accuracy", "fixed_fp", "fixed_fn", "new_fp", "new_fn"], expected_rows=5)
    write_table("table_04_dev_metric_comparison", dev_table, "Selected Dev Metrics by Ticket", "tab:dev_metrics", pred_sources,
                "Precision, recall, F1, and accuracy are unitless in [0,1]; confusion entries are tweet counts. Selection used dev evidence.",
                display_columns=["ticket", "precision_target_1", "recall_target_1", "f1_target_1", "accuracy", "true_negative", "false_positive", "false_negative", "true_positive"], expected_rows=5)
    write_table("table_05_heldout_metric_comparison", heldout_table, "Frozen Held-Out Metrics by Ticket", "tab:heldout_metrics", pred_sources,
                "Metrics are descriptive held-out results after dev-only freezing; they were not used to reopen selection.",
                display_columns=["ticket", "precision_target_1", "recall_target_1", "f1_target_1", "accuracy", "true_negative", "false_positive", "false_negative", "true_positive"], expected_rows=5)
    write_table("table_06_confusion_matrix_summary", confusion_table, "Selected Dev and Held-Out Confusion-Matrix Counts", "tab:confusion_summary", pred_sources,
                "TN/FP/FN/TP use target 1 as the positive class; counts are tweets.", expected_rows=10)
    write_table("table_07_prediction_transitions", transition_table, "Prediction Transitions Relative to the Ticket 1 Frozen Baseline", "tab:prediction_transitions", pred_sources,
                "Fixed/new FP/FN are disjoint correctness transitions; counts are tweets and are reported separately for dev and held-out.", expected_rows=10)

    # Table 8: normalization variants and robustness evidence.
    norm_metrics_path = "experiments/ticket-2/dev/results/dev_metrics.csv"
    norm_robust_path = "experiments/ticket-2/dev/robustness/robustness_metrics.csv"
    norm = load_csv(norm_metrics_path)
    norm_robust = load_csv(norm_robust_path)
    record_check("normalization variant count", len(norm) == 7, f"rows={len(norm)}")
    selected_robust = norm_robust[norm_robust["evaluated_variant"] != "raw_text_control"].copy()
    robust_lookup = selected_robust.set_index("evaluated_variant")
    norm_table = norm.copy()
    norm_table.insert(0, "split", "dev")
    norm_table["robustness_affected_rows"] = norm_table["variant"].map(robust_lookup["affected_rows"])
    norm_table["robustness_changed_predictions"] = norm_table["variant"].map(robust_lookup["changed_predictions"])
    norm_table["robustness_invariance_rate"] = norm_table["variant"].map(robust_lookup["prediction_invariance_rate"])
    write_table("table_08_normalization_experiments", norm_table, "Ticket 2 Normalization and Invariance Results on Dev", "tab:normalization",
                source(norm_metrics_path, norm_robust_path),
                "All variants use the frozen baseline model family; robustness values refer to each transformation's matched perturbation. Emoji affected zero dataset rows.",
                display_columns=["variant", "f1_target_1", "fixed_fp", "fixed_fn", "new_fp", "new_fn", "robustness_affected_rows", "robustness_changed_predictions", "robustness_invariance_rate"], expected_rows=7)

    # Table 9: shallow-feature candidates plus the robustness evidence used for rejection.
    shallow_metrics_path = "experiments/ticket-3/dev/results/dev_metrics.csv"
    shallow_robust_path = "experiments/ticket-3/dev/robustness/robustness_metrics.csv"
    shallow = load_csv(shallow_metrics_path)
    shallow.insert(0, "split", "dev")
    shallow_robust = load_csv(shallow_robust_path)
    keyword_mask = shallow_robust[shallow_robust["perturbation"] == "mask_keyword"].set_index("variant")
    neutral = shallow_robust[shallow_robust["perturbation"] == "neutralize_superficial_text"].set_index("variant")
    shallow["keyword_mask_changed_predictions"] = shallow["variant"].map(keyword_mask["changed_predictions"])
    shallow["keyword_mask_f1"] = shallow["variant"].map(keyword_mask["f1_target_1"])
    shallow["neutralized_changed_predictions"] = shallow["variant"].map(neutral["changed_predictions"])
    shallow["neutralized_f1"] = shallow["variant"].map(neutral["f1_target_1"])
    write_table("table_09_shortcut_shallow_feature_audit", shallow, "Ticket 3 Shortcut and Shallow-Feature Audit on Dev", "tab:shortcut_audit",
                source(shallow_metrics_path, shallow_robust_path),
                "Candidate metrics and matched perturbation outcomes are kept on dev. Blank robustness cells mean the perturbation was not applicable to that variant.",
                display_columns=["variant", "f1_target_1", "prediction_changes", "keyword_mask_changed_predictions", "keyword_mask_f1", "neutralized_changed_predictions", "neutralized_f1"], expected_rows=10)

    # Table 10 and Figure 1: model/decision-rule candidates and threshold sweep.
    model_path = "experiments/ticket-4/dev/results/dev_model_metrics.csv"
    threshold_path = "results/threshold_sweep.csv"
    model_table = load_csv(model_path)
    model_table.insert(0, "split", "dev")
    write_table("table_10_decision_rule_model_comparison", model_table, "Ticket 4 Model and Decision-Rule Comparison on Dev", "tab:model_comparison",
                source(model_path, threshold_path),
                "All rows preserve the frozen text representation; only the listed classifier/regularization/class-weight/threshold lever changes.",
                display_columns=["variant", "classifier_family", "class_weight", "decision_threshold", "precision_target_1", "recall_target_1", "f1_target_1", "accuracy", "fixed_fn", "new_fp"], expected_rows=8)
    sweep = load_csv(threshold_path, ["threshold", "precision_target_1", "recall_target_1", "f1_target_1"])
    record_check("threshold sweep row count", len(sweep) == 61, f"rows={len(sweep)}")
    record_check("threshold sweep grid", np.allclose(sweep["threshold"].to_numpy(), np.arange(0.20, 0.801, 0.01), atol=1e-12), "0.20--0.80 inclusive, step 0.01")
    sweep.to_csv(FIGURES / "figure_01_threshold_sweep_data.csv", index=False, float_format="%.17g")
    ASSETS.append({"asset": rel(FIGURES / "figure_01_threshold_sweep_data.csv"), "asset_type": "figure_source_csv", "title": "Ticket 4 Threshold Sweep on Dev", "split": "dev", "comparison_baseline": "Ticket 1 frozen baseline at threshold 0.50", "source_artifacts": threshold_path, "notes": "61 thresholds from 0.20 to 0.80 inclusive."})
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    colors = {"precision_target_1": "#0072B2", "recall_target_1": "#D55E00", "f1_target_1": "#009E73"}
    labels = {"precision_target_1": "Precision", "recall_target_1": "Recall", "f1_target_1": "F1"}
    for metric in colors:
        ax.plot(sweep["threshold"], sweep[metric], label=labels[metric], color=colors[metric], linewidth=1.6)
    best = sweep.loc[sweep["f1_target_1"].idxmax()]
    ax.axvline(float(best["threshold"]), color="#009E73", linestyle="--", linewidth=1, label=f"Best F1 threshold ({best['threshold']:.2f})")
    ax.axvline(0.50, color="#555555", linestyle=":", linewidth=1, label="Default threshold (0.50)")
    ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(0, 1), xlim=(0.20, 0.80), title="Ticket 4 threshold sweep on dev")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="lower center")
    save_figure(fig, "figure_01_threshold_sweep", "Ticket 4 Threshold Sweep on Dev", threshold_path,
                "Y-axis spans the full [0,1] metric range; the sweep evaluates the unweighted frozen baseline scores.")

    # Table 11 and Figure 4: data-quality evidence.
    audit_path = "results/data_quality_audit.csv"
    dev_dup_path = "experiments/ticket-5/dev/results/duplicate_summary.csv"
    held_dup_path = "experiments/ticket-5/heldout/full_duplicate_summary.csv"
    audit = load_csv(audit_path, ["id", "issue_type", "evidence", "disposition", "confidence"])
    record_check("data-quality audit row count", len(audit) == final_audit["rows"] == 64, f"rows={len(audit)}")
    record_check("data-quality audit stable IDs", audit["id"].is_unique, f"unique={audit['id'].nunique()}")
    dispositions = audit.groupby("disposition", as_index=False).agg(audited_rows=("id", "count"), mean_confidence=("confidence", "mean"))
    manifest_dispositions = pd.Series(final_audit["disposition_counts"]).sort_index()
    computed_dispositions = dispositions.set_index("disposition")["audited_rows"].sort_index()
    record_check("audit disposition counts", computed_dispositions.equals(manifest_dispositions.astype(computed_dispositions.dtype)), computed_dispositions.to_dict().__str__())
    dev_dup = load_csv(dev_dup_path).assign(scope="train_plus_dev")
    held_dup = load_csv(held_dup_path).assign(scope="full_with_heldout")
    duplicate_long = pd.concat([dev_dup, held_dup], ignore_index=True)
    disposition_long = dispositions.assign(scope="audit_disposition", relationship_type=lambda x: x["disposition"])
    disposition_long = disposition_long.rename(columns={"audited_rows": "member_rows"})
    for column in ("groups_or_pairs", "conflicting_groups_or_pairs", "cross_split_groups_or_pairs"):
        disposition_long[column] = np.nan
    quality_table = pd.concat([
        duplicate_long[["scope", "relationship_type", "groups_or_pairs", "member_rows", "conflicting_groups_or_pairs", "cross_split_groups_or_pairs"]],
        disposition_long[["scope", "relationship_type", "groups_or_pairs", "member_rows", "conflicting_groups_or_pairs", "cross_split_groups_or_pairs"]],
    ], ignore_index=True)
    write_table("table_11_data_quality_audit_summary", quality_table, "Ticket 5 Data-Quality Audit Summary", "tab:data_quality",
                source(audit_path, dev_dup_path, held_dup_path, "experiments/ticket-5/final_audit_manifest.json"),
                "Duplicate rows report groups/pairs and member tweets; disposition rows report audited tweet counts. Source and held-out labels remain unchanged.", expected_rows=10)
    disposition_data = dispositions[["disposition", "audited_rows"]].sort_values("audited_rows", ascending=False)
    disposition_data.to_csv(FIGURES / "figure_04_data_quality_dispositions_data.csv", index=False)
    ASSETS.append({"asset": rel(FIGURES / "figure_04_data_quality_dispositions_data.csv"), "asset_type": "figure_source_csv", "title": "Ticket 5 Data-Quality Audit Dispositions", "split": "mixed; stable IDs", "comparison_baseline": "not applicable", "source_artifacts": audit_path, "notes": "Counts are audited rows, not automatic relabelings."})
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.barh(disposition_data["disposition"], disposition_data["audited_rows"], color="#0072B2")
    ax.invert_yaxis(); ax.set(xlabel="Audited rows", ylabel="Disposition", xlim=(0, max(disposition_data["audited_rows"]) * 1.18), title="Ticket 5 audit dispositions")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(disposition_data["audited_rows"]): ax.text(value + 0.5, i, str(int(value)), va="center", fontsize=7)
    save_figure(fig, "figure_04_data_quality_dispositions", "Ticket 5 Data-Quality Audit Dispositions", audit_path,
                "Axis starts at zero; counts are audit dispositions and do not imply that labels were changed.")

    # Table 12: representative, balanced case set with stable IDs and verified values.
    change_cases = [
        (1, "dev", 5282, "solver boundary failure", "experiments/ticket-1/probes/changes/logreg_solver_liblinear_changes.csv", "probe", "A solver-only probe flips a correct borderline positive into a false negative."),
        (2, "heldout", 773, "normalization success", "experiments/ticket-2/heldout/heldout_changes_vs_frozen_baseline.csv", "candidate", "URL placeholdering fixes a baseline false positive."),
        (2, "heldout", 902, "normalization failure", "experiments/ticket-2/heldout/heldout_changes_vs_frozen_baseline.csv", "candidate", "URL placeholdering introduces a borderline false negative."),
        (3, "dev", 7, "shortcut success", "experiments/ticket-3/dev/changes/keyword_only_changes.csv", "candidate", "A keyword-only shortcut fixes this positive example."),
        (3, "dev", 25, "shortcut failure", "experiments/ticket-3/dev/changes/keyword_only_changes.csv", "candidate", "The same keyword-only score creates a false positive on an unrelated negative."),
        (3, "dev", 331, "shallow-feature failure", "experiments/ticket-3/dev/changes/text_plus_selected_shallow_features_changes.csv", "candidate", "The richer shallow-feature candidate turns figurative disaster language into a false positive."),
        (4, "heldout", 556, "recall success", "experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv", "candidate", "Class weighting fixes a baseline false negative."),
        (4, "heldout", 59, "precision failure", "experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv", "candidate", "Class weighting creates a false positive, illustrating the precision cost."),
        (4, "heldout", 767, "verified score conflict", "experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv", "candidate", "Machine-generated scores verify a new false positive and supersede stale narrative scores."),
        (5, "dev", 9470, "correction-probe success", "experiments/ticket-5/dev/correction_experiment/changes_vs_ticket4.csv", "candidate", "The correction probe fixes one Ticket 4 error."),
        (5, "dev", 4014, "correction-probe failure", "experiments/ticket-5/dev/correction_experiment/changes_vs_ticket4.csv", "candidate", "The same correction probe introduces a new false positive."),
    ]
    case_rows = []
    for ticket, split_name, stable_id, category, path, prefix, why in change_cases:
        frame = load_csv(path)
        row = frame[frame["id"] == stable_id]
        record_check(f"case ID {stable_id} unique in {path}", len(row) == 1, f"matches={len(row)}")
        item = row.iloc[0]
        candidate_pred = item[f"{prefix}_y_pred"]
        candidate_score = item[f"{prefix}_score"]
        truth_row = data[data["id"] == stable_id].iloc[0]
        record_check(f"case ID {stable_id} text and label", str(item["text"]) == str(truth_row["text"]) and int(item["y_true"]) == int(truth_row["target"]), path)
        case_rows.append({
            "ticket": ticket, "split": split_name, "id": stable_id, "case_type": category,
            "true_label": int(item["y_true"]), "baseline_pred": int(item["baseline_y_pred"]),
            "baseline_score": float(item["baseline_score"]), "candidate_pred": int(candidate_pred),
            "candidate_score": float(candidate_score), "transition_or_disposition": item["outcome"],
            "text": item["text"], "why_representative": why, "source_file": path,
        })
    quality_case_ids = [5760, 2619, 6220, 6223, 10795]
    final_pred = load_csv("predictions/final-heldout-predictions.csv", REQUIRED_PREDICTION_COLS).set_index("id")
    audit_index = audit.set_index("id")
    for stable_id in quality_case_ids:
        record_check(f"audit case ID {stable_id}", stable_id in audit_index.index and stable_id in data["id"].values, "stable ID present")
        audit_row = audit_index.loc[stable_id]
        data_row = data[data["id"] == stable_id].iloc[0]
        split_name = next(name for name, ids in split_sets.items() if stable_id in ids)
        prediction = final_pred.loc[stable_id] if stable_id in final_pred.index else None
        case_type = "potential annotation issue" if audit_row["disposition"] == "fix" else ("ambiguity" if audit_row["disposition"] == "ambiguous" else "validated model error")
        why = str(audit_row["evidence"])
        case_rows.append({
            "ticket": 5, "split": split_name, "id": stable_id, "case_type": case_type,
            "true_label": int(data_row["target"]), "baseline_pred": np.nan, "baseline_score": np.nan,
            "candidate_pred": int(prediction["y_pred"]) if prediction is not None else np.nan,
            "candidate_score": float(prediction["score"]) if prediction is not None else np.nan,
            "transition_or_disposition": audit_row["disposition"], "text": data_row["text"],
            "why_representative": why, "source_file": f"{audit_path}; predictions/final-heldout-predictions.csv",
        })
    cases = pd.DataFrame(case_rows)
    record_check("representative case stable IDs unique", cases["id"].is_unique, f"rows={len(cases)}")
    case_display = cases.copy(); case_display["text_excerpt"] = case_display["text"].map(shorten); case_display["reason_excerpt"] = case_display["why_representative"].map(lambda x: shorten(x, 130))
    write_table("table_12_representative_case_analysis", cases, "Representative Successes, Failures, Ambiguities, and Audit Cases", "tab:cases",
                source(*(sorted({p for p in cases["source_file"].str.split("; ").explode().tolist()}))),
                "Stable IDs, text, labels, predictions, and scores are verified against source artifacts. 'Potential annotation issue' is not treated as a confirmed mislabel.",
                display_columns=None, expected_rows=16)
    write_tex_table(case_display, TABLES / "table_12_representative_case_analysis.tex", "Representative Successes, Failures, Ambiguities, and Audit Cases", "tab:cases",
                    "The CSV preserves full text, scores, and rationales; the compact report table preserves the verified numeric case ledger. Stable IDs support exact lookup.",
                    ["ticket", "split", "id", "case_type", "true_label", "baseline_pred", "baseline_score", "candidate_pred", "candidate_score", "transition_or_disposition"])

    # Tables 13--14: verifiable project events and AI verification workflow.
    difficulty_rows = [
        {"difficulty": "Assignment reference and frozen baseline did not match", "verified_evidence": f"Held-out absolute F1 gap={comparison['absolute_difference']:.12g}; tolerance={comparison['tolerance']:.12g}.", "response": "Ran isolated dev probes and structural checks, then retained the reproducible frozen implementation and documented the unresolved gap.", "residual_limit": "The assignment's hidden reference pipeline remains unknown.", "source_file": "experiments/ticket-1/heldout/primary_contract_comparison.json; experiments/ticket-1/probes/dev_probe_metrics.csv; experiments/ticket-1/probes/configuration_audit.json"},
        {"difficulty": "Score-level numerical drift under clean replay", "verified_evidence": "Final replay audit records stable labels/order and bounded score drift.", "response": "Compared hashes, stable IDs, labels, metrics, and score differences under explicit tolerances.", "residual_limit": "Exact floating-point identity is not claimed.", "source_file": "experiments/final-reproducibility-audit/reproducibility_verification.json; experiments/final-reproducibility-audit/reproduced_summary.csv"},
        {"difficulty": "Apparent dev gain depended on shortcut-prone features", "verified_evidence": "Matched keyword masking and superficial-text neutralization materially changed predictions and F1.", "response": "Rejected the richer candidate and retained the text-only frozen baseline.", "residual_limit": "The audit covers predefined shallow features and perturbations, not every possible shortcut.", "source_file": shallow_metrics_path + "; " + shallow_robust_path + "; experiments/ticket-3/frozen_decision.json"},
        {"difficulty": "Recall gains created precision costs", "verified_evidence": "Balanced Logistic Regression fixes false negatives while introducing false positives on dev and held-out.", "response": "Evaluated bounded candidates, a full threshold sweep, confusion matrices, and transition counts before freezing the default-threshold balanced model.", "residual_limit": "The operating point reflects target-1 F1, not a deployment cost function.", "source_file": model_path + "; " + threshold_path + "; experiments/ticket-4/frozen_decision.json; experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv"},
        {"difficulty": "Duplicate conflicts and ambiguous annotations", "verified_evidence": "Duplicate summaries and 64 stable-ID audit records include conflicts and four dispositions.", "response": "Kept source/held-out labels unchanged, separated fix/ambiguous/keep/reject dispositions, and gated a train-only correction probe on dev.", "residual_limit": "Audit dispositions are evidence assessments, not ground-truth adjudications.", "source_file": audit_path + "; " + dev_dup_path + "; " + held_dup_path + "; experiments/ticket-5/final_audit_manifest.json"},
        {"difficulty": "Emoji normalization lacked dataset coverage", "verified_evidence": "The matched robustness artifact reports zero affected dev rows.", "response": "Retained the zero-coverage result and did not infer a dataset-level benefit.", "residual_limit": "The emoji hypothesis remains unresolved on this split.", "source_file": norm_robust_path},
    ]
    difficulty_table = pd.DataFrame(difficulty_rows)
    write_table("table_13_difficulties_and_solutions", difficulty_table, "Verified Project Difficulties, Responses, and Residual Limits", "tab:difficulties",
                source(*(sorted({p for p in difficulty_table["source_file"].str.split("; ").explode().tolist()}))),
                "Only difficulties supported by generated artifacts are included; narrative-only events are excluded.",
                display_columns=["difficulty", "verified_evidence", "response", "residual_limit"], expected_rows=6)
    write_wrapped_tex_table(difficulty_table, TABLES / "table_13_difficulties_and_solutions.tex",
                            "Verified Project Difficulties, Responses, and Residual Limits", "tab:difficulties",
                            "Only difficulties supported by generated artifacts are included; narrative-only events are excluded.",
                            ["difficulty", "verified_evidence", "response", "residual_limit"], [0.18, 0.22, 0.31, 0.22])

    chat_path = "logs/chat.md"
    chat_text = (ROOT / chat_path).read_text(encoding="utf-8")
    record_check("AI workflow log present", len(chat_text.strip()) > 0, f"characters={len(chat_text)}")
    ai_rows = [
        {"workflow_stage": "Evidence and contract audit", "ai_assistance": "Repository inspection, requirement decomposition, and artifact inventory", "verification": "Claims cross-checked against handout, clarifications, generated CSV/JSON, predictions, and frozen manifests", "human_accountability": "Final interpretation and submission remain the student's responsibility", "source_file": f"{chat_path}; report_evidence_audit.md; results/report_evidence_matrix.csv"},
        {"workflow_stage": "Implementation and testing", "ai_assistance": "Pipeline/test drafting and diagnostic support", "verification": "Automated tests, deterministic split checks, artifact schemas, hashes, and row-level assertions", "human_accountability": "No AI statement is accepted when it conflicts with machine-generated artifacts", "source_file": f"{chat_path}; tests; pipeline"},
        {"workflow_stage": "Experiment orchestration", "ai_assistance": "Running authorized dev experiments and producing frozen artifacts", "verification": "Selections use dev evidence; held-out completion files and frozen manifests enforce usage policy", "human_accountability": "Held-out results did not reopen ticket selection", "source_file": f"{chat_path}; {manifest_path}; experiments/ticket-1; experiments/ticket-2; experiments/ticket-3; experiments/ticket-4; experiments/ticket-5"},
        {"workflow_stage": "Reproducibility audit", "ai_assistance": "Clean-process replay and comparison of archived outputs", "verification": "Five tickets reproduced with stable IDs, labels, metrics, transitions, and bounded score tolerances", "human_accountability": "Exact score identity is not overstated", "source_file": f"{chat_path}; experiments/final-reproducibility-audit/reproducibility_verification.json; experiments/final-reproducibility-audit/reproduced_summary.csv"},
        {"workflow_stage": "Report asset generation", "ai_assistance": "Programmatic table, figure, and case-material generation", "verification": "This script recomputes metrics, verifies labels/IDs/configuration, and records an asset manifest", "human_accountability": "Generated assets must be reviewed before inclusion", "source_file": "scripts/generate_report_assets.py; report_assets/validation_report.json; report_assets/asset_manifest.csv"},
    ]
    ai_table = pd.DataFrame(ai_rows)
    write_table("table_14_ai_usage_and_verification", ai_table, "AI Usage and Verification Workflow", "tab:ai_usage",
                source(chat_path, manifest_path, "experiments/final-reproducibility-audit/reproducibility_verification.json", "experiments/final-reproducibility-audit/reproduced_summary.csv"),
                "This is a workflow declaration, not a substitute for source citations or human review.",
                display_columns=["workflow_stage", "ai_assistance", "verification", "human_accountability"], expected_rows=5)
    write_wrapped_tex_table(ai_table, TABLES / "table_14_ai_usage_and_verification.tex",
                            "AI Usage and Verification Workflow", "tab:ai_usage",
                            "This is a workflow declaration, not a substitute for source citations or human review.",
                            ["workflow_stage", "ai_assistance", "verification", "human_accountability"], [0.16, 0.25, 0.34, 0.18])

    # Figures 2--3: cross-ticket F1 and held-out transition counts, with source data.
    f1_data = all_metric_table[["ticket", "split", "f1_target_1"]].copy()
    f1_data.to_csv(FIGURES / "figure_02_dev_heldout_f1_data.csv", index=False, float_format="%.17g")
    ASSETS.append({"asset": rel(FIGURES / "figure_02_dev_heldout_f1_data.csv"), "asset_type": "figure_source_csv", "title": "Dev and Held-Out F1 by Ticket", "split": "dev and heldout", "comparison_baseline": "Ticket 1 frozen baseline", "source_artifacts": pred_sources, "notes": "F1 is recomputed from archived predictions."})
    pivot = f1_data.pivot(index="ticket", columns="split", values="f1_target_1")
    x = np.arange(len(pivot)); width = 0.36
    fig, ax = plt.subplots(figsize=(7.16, 3.1))
    ax.bar(x - width/2, pivot["dev"], width, label="Dev", color="#0072B2")
    ax.bar(x + width/2, pivot["heldout"], width, label="Held-out", color="#E69F00")
    ax.set(xticks=x, xticklabels=[f"T{t}" for t in pivot.index], ylabel="Target-1 F1", ylim=(0, 1), title="Selected dev and held-out F1 by ticket")
    ax.grid(axis="y", alpha=0.25); ax.legend()
    for xpos, values in ((x-width/2, pivot["dev"]), (x+width/2, pivot["heldout"])):
        for xi, val in zip(xpos, values): ax.text(xi, val + 0.018, f"{val:.3f}", ha="center", fontsize=6)
    save_figure(fig, "figure_02_dev_heldout_f1", "Dev and Held-Out F1 by Ticket", pred_sources,
                "Y-axis spans [0,1]; bars are descriptive and do not imply held-out selection.")

    heldout_transition_data = transition_table[transition_table["split"] == "heldout"].copy()
    heldout_transition_data.to_csv(FIGURES / "figure_03_prediction_transitions_data.csv", index=False)
    ASSETS.append({"asset": rel(FIGURES / "figure_03_prediction_transitions_data.csv"), "asset_type": "figure_source_csv", "title": "Held-Out Error Transitions Relative to Ticket 1", "split": "heldout", "comparison_baseline": "Ticket 1 frozen baseline", "source_artifacts": pred_sources, "notes": "Counts separate fixed and newly introduced error types."})
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    transition_cols = ["fixed_fp", "fixed_fn", "new_fp", "new_fn"]
    colors = ["#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
    x = np.arange(len(heldout_transition_data)); width = 0.18
    for idx, (column, color) in enumerate(zip(transition_cols, colors)):
        positions = x + (idx - 1.5) * width
        values = heldout_transition_data[column].to_numpy()
        ax.bar(positions, values, width, label=column.replace("_", " ").upper(), color=color)
    maximum = max(heldout_transition_data[transition_cols].to_numpy().max(), 1)
    ax.set(xticks=x, xticklabels=[f"T{t}" for t in heldout_transition_data["ticket"]], ylabel="Tweets", ylim=(0, maximum * 1.18), title="Held-out error transitions relative to Ticket 1")
    ax.grid(axis="y", alpha=0.25); ax.legend(ncol=4)
    save_figure(fig, "figure_03_prediction_transitions", "Held-Out Error Transitions Relative to Ticket 1", pred_sources,
                "Axis starts at zero; fixed and new FP/FN counts are shown separately without netting successes against failures.")

    # Cross-table compatibility and reproducibility assertions.
    record_check("Ticket 1 and Ticket 3 selected configurations match", predictions[(1, "heldout")][["id", "y_true", "y_pred"]].equals(predictions[(3, "heldout")][["id", "y_true", "y_pred"]]), "held-out IDs/labels/predictions identical")
    record_check("Ticket 4 and Ticket 5 selected configurations match", predictions[(4, "heldout")][["id", "y_true", "y_pred"]].equals(predictions[(5, "heldout")][["id", "y_true", "y_pred"]]), "held-out IDs/labels/predictions identical")
    reproduced = load_csv("experiments/final-reproducibility-audit/reproduced_summary.csv")
    archived = load_csv("results/summary.csv")
    record_check("reproduced summary row count", len(reproduced) == len(archived) == 5, "five tickets")
    numeric = ["dev_f1_target_1", "heldout_f1_target_1", "heldout_accuracy", "fixed_fp", "fixed_fn", "new_fp", "new_fn"]
    record_check("reproduced summary values", np.allclose(reproduced[numeric].to_numpy(float), archived[numeric].to_numpy(float), atol=1e-12), "archived and replay summaries agree")

    # Machine-readable manifest and validation report.
    manifest_frame = pd.DataFrame(ASSETS)
    manifest_frame.to_csv(OUT / "asset_manifest.csv", index=False)
    validation = {
        "status": "PASS",
        "generated_by": "scripts/generate_report_assets.py",
        "source_data_sha256": sha256(data_path),
        "split_sha256": sha256(split_path),
        "heldout_labels_modified": final_audit["heldout_labels_modified"],
        "heldout_rows_removed": final_audit["heldout_rows_removed"],
        "table_count": len(list(TABLES.glob("*.csv"))),
        "figure_count_png": len(list(FIGURES.glob("*.png"))),
        "figure_count_svg": len(list(FIGURES.glob("*.svg"))),
        "asset_manifest_rows": len(manifest_frame),
        "checks": CHECKS,
        "known_conflict": {
            "stable_id": 767,
            "status": "CONFLICTING_NARRATIVE_SUPERSEDED_BY_MACHINE_ARTIFACT",
            "machine_source": "experiments/ticket-4/heldout/heldout_changes_vs_frozen_baseline.csv",
            "baseline_score": float(cases.loc[cases["id"] == 767, "baseline_score"].iloc[0]),
            "candidate_score": float(cases.loc[cases["id"] == 767, "candidate_score"].iloc[0]),
        },
        "omitted_assets": [],
    }
    (OUT / "validation_report.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme = f"""# Verified report assets

Regenerate with:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\generate_report_assets.py
```

The generator reads only frozen or previously generated repository artifacts. It does not fit models, rerun held-out evaluation, change labels, or alter prediction files.

- Tables: `{rel(TABLES)}` (CSV source plus IEEE/LaTeX-ready snippets)
- Figures: `{rel(FIGURES)}` (PNG, SVG, and source CSV)
- Provenance: `report_assets/asset_manifest.csv`
- Assertions: `report_assets/validation_report.json`

Metric definitions: precision = TP/(TP+FP), recall = TP/(TP+FN), F1 is their harmonic mean, and accuracy = (TP+TN)/N. Target 1 is the positive class. Metrics are unitless in [0,1]; confusion and transition values are tweet counts. Every split and comparison baseline is explicit in the source tables or manifest.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "PASS", "tables": validation["table_count"], "png_figures": validation["figure_count_png"], "svg_figures": validation["figure_count_svg"], "checks": len(CHECKS), "assets": len(manifest_frame)}, indent=2))


if __name__ == "__main__":
    main()
