"""Freeze the dev-only Ticket 3 decision to reject shortcut features."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .artifacts import write_json_artifact, write_text_artifact
from .run_ticket3_heldout import sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_DIR = PROJECT_ROOT / "experiments" / "ticket-3" / "dev"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "ticket-3" / "frozen_decision.json"
SELECTED = "raw_text_tfidf_logistic_regression"
REASON = (
    "Retain the frozen text-only baseline. Although text plus keyword and selected shallow features raised dev F1 to 0.7493956486704271, "
    "its gain was shortcut-sensitive: masking keyword changed 702 predictions and reduced F1 to 0.6423057128152342, while superficial-text "
    "neutralization reduced F1 to 0.7330677290836654. Text plus keyword alone was worse than baseline (0.7350835322195705). Sparse location, "
    "length-only, and shallow-only variants were substantially weaker. The visible gain is therefore rejected rather than adopted automatically."
)


def _arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir",type=Path,default=DEFAULT_DEV_DIR)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args=_arguments()
    if args.output.exists(): raise RuntimeError("Ticket 3 freeze exists; refusing overwrite")
    heldout_dir=PROJECT_ROOT / "experiments" / "ticket-3" / "heldout"
    if heldout_dir.exists() and any(heldout_dir.iterdir()): raise RuntimeError("Ticket 3 held-out artifacts exist before freeze")
    summary=pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    if (summary["ticket"]=="ticket_3").any(): raise RuntimeError("Ticket 3 summary exists before freeze")
    metrics=pd.read_csv(args.dev_dir / "results" / "dev_metrics.csv")
    robustness=pd.read_csv(args.dev_dir / "robustness" / "robustness_metrics.csv")
    selected=metrics.loc[metrics["variant"]==SELECTED].iloc[0]
    tempting=metrics.loc[metrics["variant"]=="text_plus_selected_shallow_features"].iloc[0]
    masked=robustness.loc[(robustness["variant"]=="text_plus_selected_shallow_features")&(robustness["perturbation"]=="mask_keyword")].iloc[0]
    surface=robustness.loc[(robustness["variant"]=="text_plus_selected_shallow_features")&(robustness["perturbation"]=="neutralize_superficial_text")].iloc[0]
    command=subprocess.list2cmdline([sys.executable,"-m","pipeline.freeze_ticket3",*sys.argv[1:]])
    paths={
        "data_sha256":PROJECT_ROOT / "data" / "train.csv",
        "split_sha256":PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "ticket1_freeze_sha256":PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
        "shortcut_source_sha256":PROJECT_ROOT / "pipeline" / "shortcut_features.py",
        "dev_runner_source_sha256":PROJECT_ROOT / "pipeline" / "run_ticket3_dev.py",
        "heldout_runner_source_sha256":PROJECT_ROOT / "pipeline" / "run_ticket3_heldout.py",
        "experiment_plan_sha256":args.dev_dir / "experiment_plan.json",
        "dev_metrics_sha256":args.dev_dir / "results" / "dev_metrics.csv",
        "robustness_metrics_sha256":args.dev_dir / "robustness" / "robustness_metrics.csv",
        "text_control_dev_predictions_sha256":args.dev_dir / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv",
        "baseline_heldout_predictions_sha256":PROJECT_ROOT / "predictions" / "heldout_predictions.csv",
        "requirements_lock_sha256":PROJECT_ROOT / "requirements-lock.txt",
    }
    freeze={"ticket":3,"freeze_status":"frozen_before_ticket3_heldout_reporting","frozen_at":datetime.now().astimezone().isoformat(timespec="seconds"),"selected_variant":SELECTED,"decision":"Reject keyword, location, length, and selected shallow additions; retain the frozen text-only baseline.","decision_reason":REASON,"decision_split":"dev_ids only","prior_ticket_heldout_artifacts_exist":True,"ticket3_heldout_artifact_used_in_decision":False,"ticket3_heldout_reporting_count_at_freeze":0,"selection_reopening_permitted":False,"selected_dev_evidence":{key:(selected[key].item() if hasattr(selected[key],"item") else selected[key]) for key in ("precision_target_1","recall_target_1","f1_target_1","accuracy","true_negative","false_positive","false_negative","true_positive")},"rejected_best_visible_candidate":{"variant":"text_plus_selected_shallow_features","dev_f1_target_1":float(tempting["f1_target_1"]),"f1_delta_vs_baseline":float(tempting["f1_delta_vs_frozen_baseline"]),"fixed_fp":int(tempting["fixed_fp"]),"fixed_fn":int(tempting["fixed_fn"]),"new_fp":int(tempting["new_fp"]),"new_fn":int(tempting["new_fn"]),"keyword_masked_f1":float(masked["f1_target_1"]),"superficial_neutralized_f1":float(surface["f1_target_1"])},"heldout_reporting_mode":"Reuse the already validated Ticket 1 stable baseline predictions after freeze; do not refit or rerun the baseline on held-out.","exact_freeze_command":command,"integrity":{key:sha256(path) for key,path in paths.items()}}
    write_json_artifact(freeze,args.output)
    write_text_artifact("\n".join(["# Ticket 3 Freeze Decision","",f"Frozen at: {freeze['frozen_at']}","","Selected decision: retain the frozen raw-text baseline and reject shortcut additions.","",REASON,"","No Ticket 3 held-out artifact or metric was used in this decision. Prior-ticket held-out artifacts already existed, so the Ticket 3 report will reuse the validated baseline predictions without a new held-out fit or prediction pass."]),args.output.with_name("freeze_decision.md"))
    print(json.dumps(freeze,indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
