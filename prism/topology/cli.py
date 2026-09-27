"""Command-line interface for standalone SAM-mask topology."""

import argparse
import json
from pathlib import Path

from .core import TopologySettings, analyze_masks
from .io import load_masks, save_analysis


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sliding-window PCA and persistent H1 for ordered SAM masks."
    )
    parser.add_argument("--masks", type=Path, required=True, help="NPZ, NPY, or mask-image directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, help="Required when timestamps are absent")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--period-sec", type=float, help="Use a known period instead of auto-estimation")
    parser.add_argument("--d", type=int, default=23, help="Embedding dimension; each point uses d+1 masks")
    parser.add_argument("--n-points", type=int, default=300, help="Number of interpolated window starts")
    parser.add_argument("--min-period-sec", type=float, default=0.40)
    parser.add_argument("--max-period-sec", type=float, default=5.0)
    parser.add_argument("--min-cycles", type=float, default=2.5)
    parser.add_argument("--spectral-concentration-min", type=float, default=0.15)
    parser.add_argument(
        "--frequency-method",
        choices=("lagged_ssm", "welch_pc"),
        default="lagged_ssm",
        help="Primary period estimator (default: lagged_ssm)",
    )
    parser.add_argument("--welch-components", type=int, default=3)
    parser.add_argument("--density-k", type=int, default=20)
    parser.add_argument(
        "--density-retain-fraction",
        type=float,
        default=0.80,
        help="Densest fraction retained before sphere normalization (default: 0.80)",
    )
    parser.add_argument("--no-h1", action="store_true", help="Skip Ripser and compute PCA/distances only")
    return parser


def main():
    args = build_parser().parse_args()
    masks, timestamps, valid, metadata = load_masks(
        args.masks,
        fps=args.fps,
        threshold=args.threshold,
    )
    settings = TopologySettings(
        d=args.d,
        n_points=args.n_points,
        period_sec=args.period_sec,
        min_period_sec=args.min_period_sec,
        max_period_sec=args.max_period_sec,
        min_cycles=args.min_cycles,
        spectral_concentration_min=args.spectral_concentration_min,
        frequency_method=args.frequency_method,
        welch_components=args.welch_components,
        density_k=args.density_k,
        density_retain_fraction=args.density_retain_fraction,
        compute_h1=not args.no_h1,
    )
    result = analyze_masks(masks, timestamps, valid_mask=valid, settings=settings)
    files = save_analysis(
        result,
        args.output,
        input_metadata=metadata,
        title=args.masks.stem if args.masks.is_file() else args.masks.name,
    )
    report = {
        "period": result["summary"]["period"],
        "normalized": result["summary"]["normalized"],
        "files": files,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
