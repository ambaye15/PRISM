#!/usr/bin/env python3
import argparse
from pathlib import Path

from prism.validation.multiview.noninferiority import (
    run_noninferiority,
    summarize_noninferiority,
)
from prism.plotting.multiview_noninferiority import plot_noninferiority


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact 3D, calibrated multiview reconstruction, clear and "
            "random single views, and an unfavorable occluded view."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--keypoint-noise-px", type=float, default=1.0)
    parser.add_argument("--missing-probability", type=float, default=0.05)
    parser.add_argument("--random-views", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    frame = run_noninferiority(
        args.input,
        args.output,
        sigma_px=args.keypoint_noise_px,
        missing_probability=args.missing_probability,
        random_views=args.random_views,
        seed=args.seed,
    )
    _, summary = summarize_noninferiority(frame, args.summary_root)
    plot_noninferiority(summary, args.summary_root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
