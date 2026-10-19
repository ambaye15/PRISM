import argparse
from pathlib import Path

from prism.validation.multiview.construct_recovery import regularity_rows, regularity_summary


def main():
    parser = argparse.ArgumentParser(
        description="Test whether PS decreases with controlled temporal irregularity."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--levels", type=float, nargs="+", default=[0, .1, .2, .4])
    parser.add_argument("--keypoint-noise-px", type=float, default=1.0)
    parser.add_argument("--missing-probability", type=float, default=0.05)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    frame = regularity_rows(
        args.input,
        args.levels,
        args.keypoint_noise_px,
        args.missing_probability,
        args.seed,
        args.max_files,
    )
    summary, paired = regularity_summary(
        frame, args.output_root, args.bootstrap_replicates, args.seed
    )
    print("\nREGULARITY RECOVERY")
    print(summary.to_string(index=False))
    print("\nPAIRED VIEW COMPARISON")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
