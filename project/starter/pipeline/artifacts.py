"""Stable artifact writing and machine-checkable validation."""
import json
from pathlib import Path
import pandas as pd
from .config import RESULTS, PREDICTIONS, EXPERIMENTS, TICKETS
from .data_quality import validate_dispositions
from .metrics import binary_metrics, error_transitions

PRED_COLS=["id","y_true","y_pred","score","model_name","ticket"]
SUMMARY_COLS=["ticket","model_name","dev_f1_target_1","heldout_f1_target_1","heldout_accuracy","fixed_fp","fixed_fn","new_fp","new_fn","decision","decision_reason"]
SWEEP_COLS=["ticket","threshold","precision_target_1","recall_target_1","f1_target_1"]
AUDIT_COLS=["id","issue_type","evidence","disposition","confidence"]


def ensure_dirs():
    for p in (RESULTS,PREDICTIONS,EXPERIMENTS,TICKETS,PREDICTIONS/"dev",PREDICTIONS/"heldout"):
        p.mkdir(parents=True,exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path, sort_by: list[str] | None=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if sort_by: df=df.sort_values(sort_by,kind="stable").reset_index(drop=True)
    df.to_csv(path,index=False,float_format="%.12g",lineterminator="\n")


def write_json(value, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")


def validate_artifacts() -> dict[str,int]:
    pred=pd.read_csv(PREDICTIONS/"heldout_predictions.csv")
    summary=pd.read_csv(RESULTS/"summary.csv")
    sweep=pd.read_csv(RESULTS/"threshold_sweep.csv")
    audit=pd.read_csv(RESULTS/"data_quality_audit.csv")
    for df,cols,name in [(pred,PRED_COLS,"predictions"),(summary,SUMMARY_COLS,"summary"),(sweep,SWEEP_COLS,"sweep"),(audit,AUDIT_COLS,"audit")]:
        missing=set(cols)-set(df.columns)
        if missing: raise ValueError(f"{name} missing columns {sorted(missing)}")
    if not set(pred.y_true.unique()).issubset({0,1}) or not set(pred.y_pred.unique()).issubset({0,1}): raise ValueError("Invalid prediction labels")
    if pred.duplicated(["ticket","id"]).any(): raise ValueError("Duplicate prediction rows")
    validate_dispositions(audit)
    for row in summary.itertuples():
        part=pred[pred.ticket==row.ticket].sort_values("id")
        if len(part)!=1523: raise ValueError(f"{row.ticket}: expected 1523 predictions")
        m=binary_metrics(part.y_true,part.y_pred)
        if abs(m["f1"]-row.heldout_f1_target_1)>1e-9 or abs(m["accuracy"]-row.heldout_accuracy)>1e-9:
            raise ValueError(f"{row.ticket}: summary metrics do not reconstruct")
        baseline=pred[pred.ticket=="ticket-1"].sort_values("id")
        if not (part.id.to_numpy()==baseline.id.to_numpy()).all() or not (part.y_true.to_numpy()==baseline.y_true.to_numpy()).all():
            raise ValueError(f"{row.ticket}: predictions do not align with frozen baseline")
        transitions=error_transitions(part.id,part.y_true,baseline.y_pred,part.y_pred)
        counts=transitions.category.value_counts()
        for category in ("fixed_fp","fixed_fn","new_fp","new_fn"):
            if int(counts.get(category,0))!=int(getattr(row,category)):
                raise ValueError(f"{row.ticket}: {category} does not reconstruct from frozen baseline")
    evidence_specs={
        "ticket2_normalization_comparison.csv":({"candidate","dev_f1_target_1","selected"},12,"selected"),
        "ticket3_feature_comparison.csv":({"feature_mode","dev_f1_target_1","selected"},7,"selected"),
        "ticket4_model_grid.csv":({"model_name","family","dev_f1_target_1","selected"},15,"selected"),
        "ticket5_audit_summary.csv":({"issue_type","disposition","evidence_rows","unique_ids"},None,None),
    }
    for filename,(columns,expected_rows,selection_column) in evidence_specs.items():
        evidence=pd.read_csv(RESULTS/filename)
        missing=columns-set(evidence.columns)
        if missing: raise ValueError(f"{filename} missing columns {sorted(missing)}")
        if expected_rows is not None and len(evidence)!=expected_rows:
            raise ValueError(f"{filename}: expected {expected_rows} rows")
        if selection_column and int(evidence[selection_column].sum())!=1:
            raise ValueError(f"{filename}: expected one selected row")
    for filename in ("ticket1_probe_delta_agreement.png","ticket2_normalization_and_stress.png","ticket3_feature_and_stress.png","ticket4_model_grid.png","ticket4_dev_precision_recall.png","ticket4_dev_f1_threshold.png","ticket5_audit_distribution.png"):
        if not (RESULTS/"figures"/filename).is_file():
            raise ValueError(f"Missing evidence figure {filename}")
    return {"prediction_rows":len(pred),"summary_rows":len(summary),"sweep_rows":len(sweep),"audit_rows":len(audit)}
