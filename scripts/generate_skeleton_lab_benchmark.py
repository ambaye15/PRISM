import argparse
from pathlib import Path

from prism.validation.skeleton_lab.benchmark import generate_benchmark


def main()  :
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("output/prism_benchmark"))
    parser.add_argument("--specs-root", type=Path, default=Path("examples/specs"))
    parser.add_argument("--quick", action="store_true",
                        help="Small all-behavior smoke benchmark before the full design")
    args = parser.parse_args()
    options = {}
    if args.quick:
        options = {"frequencies": (1.0, 3.0), "durations": (2.0, 5.0, 10.0),
                   "amplitude_scales": (1.0,), "seeds": (0,)}
    manifest = generate_benchmark(args.output_root, args.specs_root, **options)
    print(f"Saved {len(manifest)} conditions to {args.output_root / 'benchmark_manifest.csv'}")


if __name__ == "__main__":
    main()
