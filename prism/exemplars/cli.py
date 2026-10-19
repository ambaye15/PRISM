import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="PRISM profile-conditioned synthetic exemplars")
    commands = parser.add_subparsers(dest="command",required=True)
    for name in ["init-reference","library","pilot","train","decode","summarize","illustrate","runtime"]:
        p = commands.add_parser(name)
        p.add_argument("--results-dir", type=Path, default=Path("results/exemplars"))
        if name in {"library","pilot","runtime"}:
            p.add_argument("--lag-profile-power", type=int, choices=(1,2), default=None)
        if name in {"library","decode"}:
            p.add_argument("--workers",type=int,default=4)
            p.add_argument("--force",action="store_true",help="Recompute cached rendered conditions")
        if name == "decode":
            p.add_argument("--retrain",action="store_true")
        if name == "illustrate":
            p.add_argument("--include-retrieval",action="store_true",
                           help="Add retrieval as a third animation column")
        if name == "runtime":
            p.add_argument("--repeats",type=int,default=3)
            p.add_argument("--child",help=argparse.SUPPRESS)
    for name in ["query","validate"]:
        p = commands.add_parser(name)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--reference",action="store_true",help="Use archived mean-distance reference assets")
        source.add_argument("--results-dir",type=Path)
        if name == "query":
            p.add_argument("--behavior",choices=("body_rocking","arm_flapping","hand_flapping"),required=True)
            p.add_argument("--profile",type=float,nargs=4,metavar=("I","E","F","PS"),required=True)
            p.add_argument("--out",type=Path,required=True)
            p.add_argument("--method",choices=("retrieval","neural","both"),default="both")
            p.add_argument("--top-k",type=int,default=3)
            p.add_argument("--max-distance",type=float,default=.35)
            p.add_argument("--no-video",action="store_true")
            p.add_argument("--seed",type=int,default=7070)
            p.add_argument("--lag-profile-power",type=int,choices=(1,2))
        else:
            p.add_argument("--out",type=Path,help="Optional path to write the validation report")
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"{exc}\nInstall optional tooling with: python -m pip install '.[exemplars]'\n")
    if result is not None:
        print(json.dumps(result,indent=None if args.command == "runtime" and args.child else 2))
    return 0


def dispatch(args):
    from .config import REFERENCE_DIR
    if args.command == "init-reference":
        from .config import initialize_reference
        return {"results_dir":str(initialize_reference(args.results_dir)),
                "flow_lag_profile_power":1,"source":"Archived reference assets"}
    if args.command == "library":
        from .experiment import run_library
        frame = run_library(args.results_dir,args.workers,args.lag_profile_power,args.force)
        if (frame.status == "error").any():
            raise ValueError(f"{int((frame.status == 'error').sum())} rendering/measurement errors; inspect records in {args.results_dir}")
        return frame.status.value_counts().to_dict()
    if args.command == "pilot":
        from .config import workspace, design, signature
        from .experiment import worker
        root, meta = workspace(args.results_dir,args.lag_profile_power)
        row = design()[0]
        row.update(condition_id="pilot",frequency_input=1.,amplitude_scale=1.,disruption=.5)
        power = meta["flow_lag_profile_power"]
        result = worker(row,root,power,signature(power))
        if result["status"] == "error":
            raise ValueError(result["error"])
        return result
    if args.command == "train":
        from .model import train_decoder
        return train_decoder(args.results_dir)
    if args.command == "decode":
        from .experiment import run_decode
        frame = run_decode(args.results_dir,args.workers,args.retrain,args.force)
        if (frame.status == "error").any():
            raise ValueError(f"{int((frame.status == 'error').sum())} decoder rendering/measurement errors; inspect records in {args.results_dir}")
        return frame.status.value_counts().to_dict()
    if args.command == "summarize":
        from .experiment import summarize
        import pandas as pd
        summarize(pd.read_csv(args.results_dir/"library_and_test.csv"),
                  pd.read_csv(args.results_dir/"decoded.csv"),args.results_dir)
    elif args.command == "query":
        from .query import query
        return query(args.behavior,args.profile,args.out,
            results_dir=REFERENCE_DIR if args.reference else args.results_dir,
            method=args.method,top_k=args.top_k,max_distance=args.max_distance,
            no_video=args.no_video,seed=args.seed,profile_power=args.lag_profile_power)
    elif args.command == "validate":
        from .validate import validate
        result = validate(REFERENCE_DIR if args.reference else args.results_dir)
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(result,indent=2))
        return result
    elif args.command == "illustrate":
        from .illustrate import illustrate
        return {"video":str(illustrate(args.results_dir,args.include_retrieval))}
    elif args.command == "runtime":
        from .runtime import probe
        return probe(args.results_dir,args.repeats,args.lag_profile_power,args.child)
