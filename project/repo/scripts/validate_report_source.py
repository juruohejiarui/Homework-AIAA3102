"""Static validation for the final IEEE report source.

This script reads existing report/evidence artifacts only. It does not run any
experiment, fit a model, or inspect held-out labels for selection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
MAIN = REPORT / "main.tex"
OUTPUT = REPORT / "source_validation.json"


def check(name: str, condition: bool, detail: str, checks: list[dict[str, str]]) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    assert condition, f"{name}: {detail}"


def balanced_braces(text: str) -> bool:
    depth = 0
    for index, character in enumerate(text):
        escaped = index > 0 and text[index - 1] == "\\" and (index < 2 or text[index - 2] != "\\")
        if escaped:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def balanced_environments(text: str) -> bool:
    stack: list[str] = []
    for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if action == "begin":
            stack.append(environment)
        elif not stack or stack.pop() != environment:
            return False
    return not stack


def main() -> None:
    text = MAIN.read_text(encoding="utf-8")
    bib_text = (REPORT / "references.bib").read_text(encoding="utf-8")
    checks: list[dict[str, str]] = []

    check("IEEE conference class", r"\documentclass[conference]{IEEEtran}" in text, "standard IEEEtran conference mode", checks)
    required_sections = [
        "Project Problem and Goal", "Methodology", "Main Evidence and Results",
        "Case Analysis", "Difficulties and Solutions", "AI Usage Declaration",
        "Discussion and Limitations", "Conclusion",
    ]
    for section in required_sections:
        check(f"required section: {section}", f"\\section{{{section}}}" in text, section, checks)
    check("abstract", r"\begin{abstract}" in text and r"\end{abstract}" in text, "abstract environment present", checks)
    check("keywords", r"\begin{IEEEkeywords}" in text and r"\end{IEEEkeywords}" in text, "IEEE keywords present", checks)
    check("appendix", r"\appendices" in text, "appendices included", checks)
    check("bibliography", r"\bibliography{references}" in text, "BibTeX bibliography included", checks)
    check("references precede appendix", text.index(r"\bibliography{references}") < text.index(r"\appendices"), "requested report order", checks)

    input_paths = re.findall(r"\\input\{([^}]+)\}", text)
    graphic_paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    check("14 generated tables included", len(input_paths) == 14, str(input_paths), checks)
    check("4 generated figures included", len(graphic_paths) == 4, str(graphic_paths), checks)
    for raw_path in input_paths + graphic_paths:
        path = (REPORT / raw_path).resolve()
        if raw_path in graphic_paths and not path.is_file():
            path = (ROOT / "report_assets/figures" / raw_path).resolve()
        check(f"included asset exists: {raw_path}", path.is_file(), str(path), checks)

    tex_sources = [MAIN] + [(REPORT / raw_path).resolve() for raw_path in input_paths]
    for path in tex_sources:
        source_text = path.read_text(encoding="utf-8")
        source_name = path.relative_to(ROOT).as_posix()
        check(f"balanced braces: {path.name}", balanced_braces(source_text), source_name, checks)
        check(f"balanced environments: {path.name}", balanced_environments(source_text), source_name, checks)
        check(f"ASCII-safe source: {path.name}", all(ord(character) < 128 for character in source_text), source_name, checks)

    combined = text
    for raw_path in input_paths:
        combined += "\n" + (REPORT / raw_path).resolve().read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{([^}]+)\}", combined))
    references = set(re.findall(r"\\ref\{([^}]+)\}", text))
    check("all cross-references resolve", references <= labels, f"unresolved={sorted(references - labels)}", checks)
    for label in sorted(labels):
        if label.startswith(("tab:", "fig:")):
            check(f"major asset referenced: {label}", label in references, label, checks)

    citation_keys = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        citation_keys.update(part.strip() for part in group.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    check("all citations resolve", citation_keys <= bib_keys, f"unresolved={sorted(citation_keys - bib_keys)}", checks)

    # Quantitative anchors are loaded from verified assets and matched to the prose.
    dataset = pd.read_csv(ROOT / "report_assets/tables/table_01_dataset_split_statistics.csv").set_index("split")
    summary = pd.read_csv(ROOT / "report_assets/tables/table_03_five_ticket_summary.csv").set_index("ticket")
    heldout = pd.read_csv(ROOT / "report_assets/tables/table_05_heldout_metric_comparison.csv").set_index("ticket")
    transitions = pd.read_csv(ROOT / "report_assets/tables/table_07_prediction_transitions.csv")
    cases = pd.read_csv(ROOT / "report_assets/tables/table_12_representative_case_analysis.csv").set_index("id")
    comparison = json.loads((ROOT / "experiments/ticket-1/heldout/primary_contract_comparison.json").read_text(encoding="utf-8"))
    replay = json.loads((ROOT / "experiments/final-reproducibility-audit/reproducibility_verification.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "experiments/ticket-5/final_audit_manifest.json").read_text(encoding="utf-8"))

    anchors = {
        "train rows": f"{int(dataset.loc['train', 'rows']):,}",
        "dev rows": f"{int(dataset.loc['dev', 'rows']):,}",
        "heldout rows": f"{int(dataset.loc['heldout', 'rows']):,}",
        "ticket 1 heldout F1": f"{heldout.loc[1, 'f1_target_1']:.6f}",
        "reference F1": f"{comparison['reference']:.6f}",
        "reference gap": f"{comparison['absolute_difference']:.6f}",
        "ticket 2 dev F1": f"{summary.loc[2, 'dev_f1']:.6f}",
        "ticket 2 heldout F1": f"{summary.loc[2, 'heldout_f1']:.6f}",
        "final heldout precision": f"{heldout.loc[5, 'precision_target_1']:.6f}",
        "final heldout recall": f"{heldout.loc[5, 'recall_target_1']:.6f}",
        "final heldout F1": f"{heldout.loc[5, 'f1_target_1']:.6f}",
        "final heldout accuracy": f"{heldout.loc[5, 'accuracy']:.6f}",
        "ID 767 baseline score": f"{cases.loc[767, 'baseline_score']:.6f}",
        "ID 767 candidate score": f"{cases.loc[767, 'candidate_score']:.6f}",
    }
    for name, value in anchors.items():
        check(f"quantitative anchor: {name}", value in text, value, checks)

    final_transition = transitions[(transitions["ticket"] == 5) & (transitions["split"] == "heldout")].iloc[0]
    check("final transition values present", all(str(int(final_transition[key])) in text for key in ("fixed_fn", "new_fp")), final_transition.to_dict().__str__(), checks)
    check("machine ID 767 values used", "0.487623" in text and "0.544815" in text, "machine CSV values", checks)
    check("stale ID 767 values excluded", "0.481794" not in text and "0.544153" not in text, "stale narrative values absent", checks)
    check("heldout integrity stated", audit["heldout_labels_modified"] is False and audit["heldout_rows_removed"] == 0 and "no source or held-out label was changed" in text, "no held-out mutation", checks)
    coefficient, exponent = f"{replay['maximum_score_difference']:.2e}".split("e")
    drift_latex = rf"{coefficient}\times10^{{{int(exponent)}}}"
    check("replay drift stated", drift_latex in text, drift_latex, checks)
    check("final model preserved", "Ticket 5: the Ticket 4 balanced Logistic Regression recipe with no training-label corrections" in text, "no post-hoc Ticket 2 selection", checks)
    check("verified author names", "LAI Jiaxing" in text and "HE Jiarui" in text, "both supplied names present", checks)
    placeholder_patterns = ("[Author", "[Department", "[City", "[Verified", "TODO", "TBD", "PLACEHOLDER")
    check("no report placeholders", not any(pattern in text for pattern in placeholder_patterns), "no unresolved placeholder tokens", checks)
    check("affiliation not fabricated", "\\IEEEauthorblockA" not in text, "unsupplied affiliation/email omitted", checks)

    word_count = len(re.findall(r"\b[\w'-]+\b", re.sub(r"%.*", "", text)))
    check("substantive report length", word_count >= 4500, f"approximate source word count={word_count}", checks)

    result = {
        "status": "PASS",
        "validation_scope": "static final report-source validation; PDF checks are recorded separately",
        "report_source": "report/main.tex",
        "references": "report/references.bib",
        "table_inputs": input_paths,
        "figure_inputs": graphic_paths,
        "citation_keys": sorted(citation_keys),
        "checks": checks,
        "remaining_warnings": [
            "affiliation, location, and email were not supplied and are omitted",
            "course-internal references have repository filenames but no public publication metadata",
            "the Ticket 1 reference discrepancy remains causally unresolved",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "word_count": word_count, "tables": len(input_paths), "figures": len(graphic_paths)}, indent=2))


if __name__ == "__main__":
    main()
