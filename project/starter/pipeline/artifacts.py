"""Stable artifact writing and machine-checkable validation."""
import json
from pathlib import Path
import pandas as pd
from .config import RESULTS, PREDICTIONS, EXPERIMENTS, TICKETS
from .data_quality import validate_dispositions
from .metrics import binary_metrics

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
        part=pred[pred.ticket==row.ticket]
        if len(part)!=1523: raise ValueError(f"{row.ticket}: expected 1523 predictions")
        m=binary_metrics(part.y_true,part.y_pred)
        if abs(m["f1"]-row.heldout_f1_target_1)>1e-9 or abs(m["accuracy"]-row.heldout_accuracy)>1e-9:
            raise ValueError(f"{row.ticket}: summary metrics do not reconstruct")
    return {"prediction_rows":len(pred),"summary_rows":len(summary),"sweep_rows":len(sweep),"audit_rows":len(audit)}
