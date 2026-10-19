import argparse
from pathlib import Path

from prism.validation.multiview.construct_recovery import run_construct_recovery


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 2D and multiview PRISM against controlled constructs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--keypoint-noise-px", type=float, default=1.0)
    parser.add_argument("--missing-probability", type=float, default=0.05)
    parser.add_argument("--random-views", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    _, _, summary, paired = run_construct_recovery(
        args.input,
        args.output,
        args.summary_root,
        args.keypoint_noise_px,
        args.missing_probability,
        args.random_views,
        args.bootstrap_replicates,
        args.seed,
    )
    print("\nCONSTRUCT RECOVERY")
    print(summary.to_string(index=False))
    print("\nPAIRED VIEW COMPARISON")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
