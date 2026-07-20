"""Command-line interface for validation, experiments, and artifact checks."""
import argparse, json
from .artifacts import validate_artifacts
from .config import EXPERIMENTS
from .data import get_splits
from .experiments import run_all


def build_parser():
    parser=argparse.ArgumentParser(prog="python -m pipeline.cli")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("validate-data")
    sub.add_parser("run-all")
    run=sub.add_parser("run-ticket"); run.add_argument("--ticket",type=int,choices=range(1,6),required=True); run.add_argument("--split",choices=["dev","heldout"],default="dev")
    freeze=sub.add_parser("freeze-ticket"); freeze.add_argument("--ticket",type=int,choices=range(1,6),required=True)
    sub.add_parser("validate-artifacts")
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=="validate-data":
        parts=get_splits(); print(json.dumps({k:{"rows":len(v),"positives":int(v.target.sum())} for k,v in parts.items()},sort_keys=True)); return 0
    if args.command=="run-all":
        result=run_all(); print(result["summary"].to_string(index=False)); return 0
    if args.command=="validate-artifacts": print(json.dumps(validate_artifacts(),sort_keys=True)); return 0
    path=EXPERIMENTS/"decisions.json"
    if args.command=="freeze-ticket":
        if not path.exists(): raise SystemExit("No dev decision exists. Run the ticket dev experiment first.")
        record=json.loads(path.read_text(encoding="utf-8")).get(f"ticket-{args.ticket}")
        if not record: raise SystemExit("Ticket decision missing")
        print(f"ticket-{args.ticket} already frozen: {record['run_id']}"); return 0
    if args.split=="heldout":
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get(f"ticket-{args.ticket}",{}).get("status")!="frozen":
            raise SystemExit("Held-out evaluation requires a frozen decision")
    # Ticket runs are idempotent full-registry rebuilds so shared baselines stay consistent.
    run_all(); print(f"ticket-{args.ticket} {args.split} artifacts regenerated"); return 0


if __name__=="__main__": raise SystemExit(main())

