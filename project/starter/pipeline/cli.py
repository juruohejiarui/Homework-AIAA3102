"""Command-line interface for validation, experiments, and artifact checks."""
import argparse, json
from .artifacts import validate_artifacts
from .data import get_splits
from .experiments import freeze_ticket, run_all, run_ticket_dev, run_ticket_heldout


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
    if args.command=="freeze-ticket":
        try: record=freeze_ticket(args.ticket)
        except ValueError as exc: raise SystemExit(str(exc)) from exc
        print(f"ticket-{args.ticket} frozen: {record['run_id']}"); return 0
    if args.split=="dev":
        record=run_ticket_dev(args.ticket)
        print(f"ticket-{args.ticket} dev decision pending: {record['run_id']}"); return 0
    try: result=run_ticket_heldout(args.ticket)
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    print(json.dumps(result,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())

