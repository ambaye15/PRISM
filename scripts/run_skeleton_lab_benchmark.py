#!/usr/bin/env python3

import argparse
from pathlib import Path

from prism.validation.skeleton_lab.benchmark import run_benchmark


def main()  :
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dimension", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=23)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--only-family", choices=("construct", "robustness"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--camera-correction", choices=("none", "synthetic", "ransac"),
                        default="none")
    parser.add_argument("--only-nuisance", nargs="+",
                        help="Run only named nuisances, e.g. clean camera_shake_sigma_px")
    args = parser.parse_args()
    result = run_benchmark(args.manifest, args.output, args.max_dimension,
                           args.dimension, args.n_points, args.only_family,
                           not args.no_resume, args.camera_correction,
                           tuple(args.only_nuisance) if args.only_nuisance else None)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
