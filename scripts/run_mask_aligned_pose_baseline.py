#!/usr/bin/env python3
"""Run a pose baseline on crops derived from the frozen DAM4SAM masks."""

import argparse

from prism.validation.pose_baseline import run_pose_baseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask-aligned pose inference and PRISM agreement analysis."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument(
        "--videos-root",
        default=None,
        help="Optional current video directory used to relocate stale absolute source paths.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default="yolo11m-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--padding-fraction", type=float, default=0.15)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--maximum-interpolation-gap", type=int, default=2)
    parser.add_argument("--minimum-valid-pair-fraction", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    merged, agreement, availability = run_pose_baseline(
        args.features,
        args.runs_root,
        args.output_root,
        videos_root=args.videos_root,
        model=args.model,
        device=args.device,
        image_size=args.image_size,
        padding_fraction=args.padding_fraction,
        confidence_threshold=args.confidence_threshold,
        maximum_interpolation_gap=args.maximum_interpolation_gap,
        minimum_valid_pair_fraction=args.minimum_valid_pair_fraction,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        force=args.force,
        limit=args.limit,
    )
    print("\nAVAILABILITY")
    print(availability.to_string(index=False))
    print("\nAGREEMENT")
    if len(agreement):
        print(agreement.to_string(index=False))
    else:
        print("No eligible paired pose/flow comparisons were available.")
        if "pose_error" in merged:
            print("\nPOSE ERRORS")
            print(merged["pose_error"].fillna("<none>").value_counts().to_string())
        diagnostics = [
            "episode_id",
            "pose_detected_frame_fraction",
            "pose_confident_landmarks_per_frame_median",
            "pose_valid_pair_fraction",
            "pose_torso_valid_pair_fraction",
            "pose_error",
        ]
        diagnostics = [column for column in diagnostics if column in merged]
        print("\nBOUT DIAGNOSTICS")
        print(merged[diagnostics].to_string(index=False))
    print("\nOutputs: " + args.output_root)


if __name__ == "__main__":
    main()

