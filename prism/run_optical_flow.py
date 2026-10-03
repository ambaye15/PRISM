"""Batch runner for optical flow inside previously saved SAM2 masks."""

import argparse
import csv
import json
from pathlib import Path

from .optical_flow import (
    OpticalFlowSettings,
    analyze_optical_flow_episode,
    make_flow_estimator,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure body-scale-normalized visual motion inside completed "
            "SAM2 subject tracks."
        )
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path("prism/runs_block_filtered")
    )
    parser.add_argument(
        "--episodes-root", type=Path, default=Path("annotated_runs/episode_runs")
    )
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--backend",
        choices=("dis", "farneback", "raft_small", "raft_large"),
        default="dis",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--camera-correction",
        choices=(
            "none",
            "translation",
            "ransac_similarity",
            "ransac_affine",
        ),
        default="ransac_similarity",
    )
    parser.add_argument(
        "--region-mode",
        choices=("source", "intersection", "flow_consistent"),
        default="flow_consistent",
    )
    parser.add_argument("--erosion-body-fraction", type=float, default=0.005)
    parser.add_argument(
        "--target-dilation-body-fraction", type=float, default=0.020
    )
    parser.add_argument(
        "--active-threshold-body-heights-per-sec", type=float, default=0.05
    )
    parser.add_argument("--minimum-valid-pixels", type=int, default=100)
    parser.add_argument(
        "--minimum-retained-pixel-fraction", type=float, default=0.20
    )
    parser.add_argument(
        "--minimum-valid-pair-fraction", type=float, default=0.80
    )
    parser.add_argument("--maximum-flow-dimension", type=int, default=768)
    parser.add_argument("--flow-preblur-sigma", type=float, default=0.75)
    parser.add_argument("--body-height-smoothing-window", type=int, default=5)
    parser.add_argument(
        "--no-forward-backward-check",
        action="store_true",
        help="Disable forward--backward optical-flow consistency filtering.",
    )
    parser.add_argument(
        "--forward-backward-threshold-body-fraction", type=float, default=0.02
    )
    parser.add_argument(
        "--ransac-subject-exclusion-body-fraction", type=float, default=0.05
    )
    parser.add_argument("--ransac-max-corners", type=int, default=1000)
    parser.add_argument("--ransac-quality-level", type=float, default=0.01)
    parser.add_argument(
        "--ransac-min-distance-pixels", type=float, default=8.0
    )
    parser.add_argument(
        "--ransac-reprojection-threshold-pixels", type=float, default=3.0
    )
    parser.add_argument("--ransac-min-matches", type=int, default=30)
    parser.add_argument(
        "--ransac-min-inlier-fraction", type=float, default=0.35
    )
    parser.add_argument(
        "--no-torso-correction",
        action="store_true",
        help="Disable subtraction of median central torso-core translation.",
    )
    parser.add_argument("--torso-radius-x-fraction", type=float, default=0.18)
    parser.add_argument("--torso-radius-y-fraction", type=float, default=0.22)
    parser.add_argument("--torso-minimum-pixels", type=int, default=25)
    parser.add_argument("--flow-state-grid-height", type=int, default=48)
    parser.add_argument("--flow-state-grid-width", type=int, default=32)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _base_feature_table(runs_root):
    candidates = (
        runs_root / "subject_dino_mask_topology_bout_features.csv",
        runs_root / "subject_dino_mask_recurrence_bout_features.csv",
        runs_root / "subject_dino_bout_features.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _identifiers(summary, episode_id):
    return {
        "episode_id": episode_id,
        "fps": summary.get("fps"),
        **(summary.get("episode") or {}),
    }


def _join_feature_tables(base_path, flow_rows, output_path):
    base_rows = _read_csv(base_path) if base_path is not None else []
    base_by_episode = {
        str(row.get("episode_id")): row
        for row in base_rows
        if row.get("episode_id") is not None
    }
    flow_by_episode = {
        str(row.get("episode_id")): row
        for row in flow_rows
        if row.get("episode_id") is not None
    }
    order = [
        str(row.get("episode_id"))
        for row in base_rows
        if row.get("episode_id") is not None
    ]
    order.extend(
        episode_id
        for episode_id in flow_by_episode
        if episode_id not in base_by_episode
    )
    joined = []
    for episode_id in order:
        if episode_id not in flow_by_episode:
            continue
        joined.append(
            {
                **base_by_episode.get(episode_id, {}),
                **flow_by_episode[episode_id],
            }
        )
    _write_csv(output_path, joined)
    return len(joined)


def main():
    args = build_parser().parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    episodes_root = args.episodes_root.expanduser().resolve()
    requested = set(args.episode_id)
    episode_dirs = sorted(
        path
        for path in runs_root.iterdir()
        if path.is_dir()
        and (path / "sam2_masks_fixed_crop.npz").is_file()
        and (not requested or path.name in requested)
    )
    found = {path.name for path in episode_dirs}
    missing = sorted(requested - found)
    if missing:
        raise FileNotFoundError(
            "No completed SAM2 run found for: " + ", ".join(missing)
        )
    if args.limit is not None:
        episode_dirs = episode_dirs[: args.limit]
    if not episode_dirs:
        raise SystemExit("No completed SAM2 runs found in %s" % runs_root)

    settings = OpticalFlowSettings(
        backend=args.backend,
        device=args.device,
        camera_correction=args.camera_correction,
        region_mode=args.region_mode,
        erosion_body_fraction=args.erosion_body_fraction,
        target_dilation_body_fraction=args.target_dilation_body_fraction,
        active_threshold_body_heights_per_sec=(
            args.active_threshold_body_heights_per_sec
        ),
        minimum_valid_pixels=args.minimum_valid_pixels,
        minimum_retained_pixel_fraction=args.minimum_retained_pixel_fraction,
        minimum_valid_pair_fraction=args.minimum_valid_pair_fraction,
        maximum_flow_dimension=args.maximum_flow_dimension,
        flow_preblur_sigma=args.flow_preblur_sigma,
        body_height_smoothing_window=args.body_height_smoothing_window,
        forward_backward_check=not args.no_forward_backward_check,
        forward_backward_threshold_body_fraction=(
            args.forward_backward_threshold_body_fraction
        ),
        ransac_subject_exclusion_body_fraction=(
            args.ransac_subject_exclusion_body_fraction
        ),
        ransac_max_corners=args.ransac_max_corners,
        ransac_quality_level=args.ransac_quality_level,
        ransac_min_distance_pixels=args.ransac_min_distance_pixels,
        ransac_reprojection_threshold_pixels=(
            args.ransac_reprojection_threshold_pixels
        ),
        ransac_min_matches=args.ransac_min_matches,
        ransac_min_inlier_fraction=args.ransac_min_inlier_fraction,
        torso_correction=not args.no_torso_correction,
        torso_radius_x_fraction=args.torso_radius_x_fraction,
        torso_radius_y_fraction=args.torso_radius_y_fraction,
        torso_minimum_pixels=args.torso_minimum_pixels,
        flow_state_grid_height=args.flow_state_grid_height,
        flow_state_grid_width=args.flow_state_grid_width,
    )
    estimator = make_flow_estimator(settings)
    rows = []
    failures = []
    reused = 0
    completed = 0
    for index, episode_dir in enumerate(episode_dirs, start=1):
        print("[%d/%d] %s" % (index, len(episode_dirs), episode_dir.name))
        summary_path = episode_dir / "optical_flow_summary.json"
        existing = {}
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        will_reuse = bool(
            not args.force
            and existing.get("schema_version") == "prism_masked_optical_flow.v3"
            and existing.get("settings") == {
                name: getattr(settings, name)
                for name in settings.__record_fields__
            }
            and (episode_dir / "optical_flow_states.npz").is_file()
        )
        try:
            summary = analyze_optical_flow_episode(
                episode_dir,
                settings=settings,
                estimator=estimator,
                episodes_root=episodes_root,
                create_figure=not args.no_figures,
                create_video=args.save_videos,
                force=args.force,
            )
            identifiers = _identifiers(summary, episode_dir.name)
            rows.append(
                {
                    **identifiers,
                    **(summary.get("measurements") or {}),
                }
            )
            if will_reuse:
                reused += 1
            else:
                completed += 1
        except Exception as error:
            failure = {
                "episode_id": episode_dir.name,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            print("  FAILED: %s: %s" % (failure["error_type"], failure["error"]))
            if args.fail_fast:
                raise

    flow_path = runs_root / "optical_flow_bout_features.csv"
    failure_path = runs_root / "optical_flow_failures.csv"
    _write_csv(flow_path, rows)
    _write_csv(failure_path, failures)

    base_path = _base_feature_table(runs_root)
    if base_path is None:
        combined_path = runs_root / "optical_flow_combined_bout_features.csv"
    else:
        combined_path = runs_root / (
            base_path.stem + "_optical_flow.csv"
        )
    n_joined = _join_feature_tables(base_path, rows, combined_path)
    report = {
        "schema_version": "prism_masked_optical_flow_batch.v2",
        "n_selected": len(episode_dirs),
        "n_completed": completed,
        "n_reused": reused,
        "n_failed": len(failures),
        "n_joined": n_joined,
        "backend": settings.backend,
        "runs_root": str(runs_root),
        "optical_flow_features": str(flow_path),
        "combined_features": str(combined_path),
        "failures": str(failure_path),
    }
    (runs_root / "optical_flow_batch_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
