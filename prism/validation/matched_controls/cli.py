import argparse
import json
from pathlib import Path

from .eulerian import LEVELS
from .experiment import run_compensation, run_eulerian, run_experiment
from .report import summarize


def main(argv=None):
    parser = argparse.ArgumentParser(prog='prism-matched-controls')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['run', 'eulerian', 'compensate', 'summarize']:
        command = commands.add_parser(name)
        command.add_argument('--results-dir', type=Path, default=Path('runs/matched_controls'))
        if name in ['run', 'eulerian']:
            command.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
            command.add_argument('--levels', type=float, nargs='+', default=list(LEVELS))
            command.add_argument('--save-grams', action='store_true')
        if name in ['run', 'eulerian', 'compensate']:
            command.add_argument('--workers', type=int, default=1)
            command.add_argument('--force', action='store_true')
        if name in ['eulerian', 'compensate']:
            command.add_argument('--shard', type=int, default=0)
            command.add_argument('--shards', type=int, default=1)
        if name in ['run', 'summarize']:
            command.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            result = run_experiment(args.results_dir, args.seeds, args.levels, args.workers,
                                    args.force, args.save_grams, not args.no_plots)
        elif args.command == 'eulerian':
            frame = run_eulerian(args.results_dir, args.seeds, args.levels, args.workers,
                                 args.shard, args.shards, args.force, args.save_grams)
            result = dict(eulerian_rows_processed=len(frame))
        elif args.command == 'compensate':
            frame = run_compensation(args.results_dir, args.workers, args.shard, args.shards, args.force)
            result = dict(compensation_rows_processed=len(frame))
        else:
            result = summarize(args.results_dir, plots=not args.no_plots)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, indent=2))
    return 0
