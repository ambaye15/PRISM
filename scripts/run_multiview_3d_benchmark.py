#!/usr/bin/env python3
import argparse
from pathlib import Path

from prism.validation.multiview.benchmark import run_benchmark
from prism.plotting.multiview import plot_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--keypoint-noise-px", type=float, default=1.0)
    parser.add_argument("--missing-probability", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    frame = run_benchmark(
        args.input,
        args.output,
        sigma_px=args.keypoint_noise_px,
        missing_probability=args.missing_probability,
        seed=args.seed,
    )
    summary = plot_results(frame, args.figures)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

