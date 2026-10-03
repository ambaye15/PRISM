"""Batch PRISM topology: flow frequency/periodicity plus mask RMS extent."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ._records import asdict
from .topology import TopologySettings, analyze_flow_states, framewise_rms_extent


SCHEMA = "prism_flow_topology.v3"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate frequency and periodicity from camera-corrected total "
            "signed flow, with framewise RMS binary-mask orbit extent."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("prism/runs_block_filtered"))
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--period-sec", type=float)
    parser.add_argument("--d", type=int, default=23)
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--min-period-sec", type=float, default=0.40)
    parser.add_argument("--max-period-sec", type=float, default=5.0)
    parser.add_argument("--min-cycles", type=float, default=2.5)
    parser.add_argument("--spectral-concentration-min", type=float, default=0.15)
    parser.add_argument("--density-k", type=int, default=20)
    parser.add_argument("--density-retain-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-valid-mask-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-extent-frames", type=int, default=8)
    parser.add_argument("--flow-clip-mad", type=float, default=6.0)
    parser.add_argument("--periodicity-score-min", type=float)
    parser.add_argument("--no-h1", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def identifiers(source, episode_id):
    return {"episode_id": episode_id, "fps": source.get("fps"), **(source.get("episode") or {})}


def extent_measurements(
    extent_result,
    valid_fraction,
    minimum_frames,
    minimum_valid_fraction,
):
    reliable = bool(
        extent_result.get("extent_valid_frames", 0) >= minimum_frames
        and valid_fraction >= minimum_valid_fraction
    )
    return {
        "mask_frame_rms_orbit_radius": extent_result.get("extent_frame_rms"),
        "mask_frame_rms_valid_frames": extent_result.get("extent_valid_frames"),
        "mask_frame_rms_valid_frame_fraction": extent_result.get(
            "extent_valid_frame_fraction"
        ),
        "mask_extent_representation": "framewise_binary_rms_orbit",
        "mask_extent_measurement_reliable": reliable,
    }


def flow_topology_measurements(flow_result, valid_fraction, settings, score_min):
    flow = flow_result["summary"]
    period = flow["period"]
    normalized = flow["normalized"]
    period_reliable = bool(
        period.get("reliable") and valid_fraction >= settings.minimum_valid_mask_fraction
    )
    score = normalized.get("periodicity_score")
    topology_reliable = bool(
        period_reliable
        and score is not None
        and (score_min is None or float(score) >= score_min)
    )
    cycle_distance = flow_result.get("cycle_return_distance")
    return {
        "topo_flow_period_sec": period.get("period_sec"),
        "topo_flow_frequency_hz": period.get("frequency_hz"),
        "topo_flow_spectral_concentration": period.get("spectral_concentration"),
        "topo_flow_estimated_cycles": period.get("estimated_cycles"),
        "topo_flow_period_reliable": period_reliable,
        "topo_flow_temporal_measurement_reliable": topology_reliable,
        "topo_flow_valid_mask_fraction": valid_fraction,
        "topo_flow_embedding_dimension": settings.d,
        "topo_flow_window_samples": settings.d + 1,
        "topo_flow_n_states": flow["embedding"].get("n_states"),
        "topo_flow_n_windows": flow["embedding"].get("n_windows_retained"),
        "flow_normalized_h1_computed": bool(normalized.get("h1_computed", False)),
        "flow_normalized_h1_max_persistence": normalized.get("h1_max_persistence"),
        "flow_normalized_h1_total_persistence": normalized.get("h1_total_persistence"),
        "flow_normalized_h1_relative_persistence": normalized.get("h1_relative_persistence"),
        "flow_normalized_periodicity_score": score,
        "flow_normalized_cycle_return_distance": cycle_distance,
        "flow_cycle_consistency": None if cycle_distance is None else float(np.clip(1.0 - cycle_distance / 2.0, 0.0, 1.0)),
    }


def _save_flow_analysis(result, output):
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "flow_topology_summary.json"
    arrays_path = output / "flow_topology_arrays.npz"
    summary_path.write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays = {key: value for key, value in result.items() if key != "summary" and value is not None}
    np.savez_compressed(arrays_path, **arrays)
    return {"summary": str(summary_path), "arrays": str(arrays_path)}


def expected_unavailability(stage, error):
    if not isinstance(error, ValueError):
        return False
    message = str(error).lower()
    if stage == "extent":
        return "fewer than" in message and "masks remain for extent" in message
    phrases = (
        "too short for lagged-ssm frequency",
        "no uninterrupted run contains at least",
        "too short for period-adapted windows",
        "states must have shape (time, height, width, 2)",
        "fewer than",
    )
    return any(phrase in message for phrase in phrases)


def issue_record(stage, error):
    return {
        "stage": stage,
        "severity": (
            "unavailable" if expected_unavailability(stage, error) else "failure"
        ),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def analyze_episode(
    episode_dir,
    settings,
    score_min,
    flow_clip_mad,
    minimum_extent_frames,
):
    mask_path = episode_dir / "sam2_masks_fixed_crop.npz"
    flow_path = episode_dir / "optical_flow_states.npz"
    with np.load(mask_path, allow_pickle=False) as saved:
        masks = saved["masks"].astype(bool)
        core = saved["core_mask"].astype(bool)
        observed = masks.reshape(len(masks), -1).any(axis=1)
        valid = saved["mask_valid"].astype(bool) if "mask_valid" in saved.files else observed

    core_count = int(np.count_nonzero(core))
    valid_fraction = float(np.count_nonzero(valid & core) / core_count) if core_count else 0.0

    output = episode_dir / "flow_topology"
    output.mkdir(parents=True, exist_ok=True)
    measurements = {}
    issues = []
    files = {}

    try:
        extent_result = framewise_rms_extent(
            masks,
            valid_mask=valid & core,
            minimum_frames=minimum_extent_frames,
        )
        measurements.update(
            extent_measurements(
                extent_result,
                valid_fraction,
                minimum_extent_frames,
                settings.minimum_valid_mask_fraction,
            )
        )
    except Exception as error:
        extent_result = None
        issues.append(issue_record("extent", error))

    extent_path = output / "mask_framewise_extent.json"
    if extent_result is not None:
        extent_path.write_text(
            json.dumps(
                {
                    "source": str(mask_path),
                    "representation": "framewise binary-mask RMS orbit radius",
                    "period_dependent": False,
                    "core_valid_mask_fraction": valid_fraction,
                    **extent_result,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        files["mask_extent"] = {"summary": str(extent_path)}

    try:
        if not flow_path.is_file():
            raise FileNotFoundError(
                "%s is missing; rerun prism.run_optical_flow first" % flow_path
            )
        with np.load(flow_path, allow_pickle=False) as saved:
            states = saved["total"].astype(np.float64)
            confidence = saved["confidence"].astype(np.float64)
            state_times = saved["timestamps_sec"].astype(np.float64)
            pair_indices = saved["pair_indices"].astype(np.int64)
        flow_result = analyze_flow_states(
            states,
            confidence,
            state_times,
            pair_indices=pair_indices,
            settings=settings,
            clip_mad=flow_clip_mad,
        )
        measurements.update(
            flow_topology_measurements(
                flow_result,
                valid_fraction,
                settings,
                score_min,
            )
        )
        files["flow"] = _save_flow_analysis(flow_result, output)
    except Exception as error:
        issues.append(issue_record("flow_topology", error))

    payload = {
        "schema_version": SCHEMA,
        "episode_id": episode_dir.name,
        "settings": {
            **asdict(settings),
            "minimum_valid_mask_fraction": settings.minimum_valid_mask_fraction,
            "periodicity_score_min": score_min,
            "flow_clip_mad": flow_clip_mad,
            "minimum_extent_frames": minimum_extent_frames,
        },
        "representations": {
            "frequency": "camera-corrected total signed flow",
            "periodicity": "sphere-normalized total signed-flow windows",
            "extent": "RMS orbit radius of framewise binary masks",
        },
        "measurements": measurements,
        "issues": issues,
        "files": files,
    }
    (episode_dir / "flow_topology_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main():
    args = build_parser().parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    requested = set(args.episode_id)
    episode_dirs = sorted(
        path for path in runs_root.iterdir()
        if path.is_dir()
        and (path / "sam2_masks_fixed_crop.npz").is_file()
        and (not requested or path.name in requested)
    )
    missing = sorted(requested.difference(path.name for path in episode_dirs))
    if missing:
        raise FileNotFoundError("no completed mask run found for: " + ", ".join(missing))
    if args.limit is not None:
        episode_dirs = episode_dirs[:args.limit]

    settings = TopologySettings(
        d=args.d,
        n_points=args.n_points,
        period_sec=args.period_sec,
        min_period_sec=args.min_period_sec,
        max_period_sec=args.max_period_sec,
        min_cycles=args.min_cycles,
        spectral_concentration_min=args.spectral_concentration_min,
        frequency_method="lagged_ssm",
        density_k=args.density_k,
        density_retain_fraction=args.density_retain_fraction,
        compute_h1=not args.no_h1,
    )
    settings.minimum_valid_mask_fraction = args.minimum_valid_mask_fraction
    expected_settings = {
        **asdict(settings),
        "minimum_valid_mask_fraction": args.minimum_valid_mask_fraction,
        "periodicity_score_min": args.periodicity_score_min,
        "flow_clip_mad": args.flow_clip_mad,
        "minimum_extent_frames": args.minimum_extent_frames,
    }
    rows = []
    failures = []
    unavailable = []
    completed = 0
    reused = 0
    for index, episode_dir in enumerate(episode_dirs, start=1):
        print("[%d/%d] %s" % (index, len(episode_dirs), episode_dir.name))
        source = read_json(episode_dir / "subject_dino_summary.json")
        summary_path = episode_dir / "flow_topology_summary.json"
        try:
            existing = read_json(summary_path)
            if not args.force and existing.get("schema_version") == SCHEMA and existing.get("settings") == expected_settings:
                topology = existing
                reused += 1
            else:
                topology = analyze_episode(
                    episode_dir,
                    settings,
                    args.periodicity_score_min,
                    args.flow_clip_mad,
                    args.minimum_extent_frames,
                )
                completed += 1
            rows.append({**identifiers(source, episode_dir.name), **(topology.get("measurements") or {})})
            for issue in topology.get("issues") or []:
                record = {"episode_id": episode_dir.name, **issue}
                if issue.get("severity") == "unavailable":
                    unavailable.append(record)
                    print(
                        "  UNAVAILABLE [%s]: %s"
                        % (issue.get("stage"), issue.get("error"))
                    )
                else:
                    failures.append(record)
                    print(
                        "  FAILED [%s]: %s"
                        % (issue.get("stage"), issue.get("error"))
                    )
        except Exception as error:
            failures.append({
                "episode_id": episode_dir.name,
                "stage": "episode",
                "severity": "failure",
                "error_type": type(error).__name__,
                "error": str(error),
            })
            print("  FAILED: %s: %s" % (type(error).__name__, error))
            if args.fail_fast:
                raise

    feature_path = runs_root / "flow_topology_bout_features.csv"
    write_csv(feature_path, rows)
    write_csv(runs_root / "flow_topology_unavailable.csv", unavailable)
    write_csv(runs_root / "flow_topology_failures.csv", failures)
    report = {
        "schema_version": "prism_flow_topology_batch.v2",
        "n_selected": len(episode_dirs),
        "n_completed": completed,
        "n_reused": reused,
        "n_unavailable": len(unavailable),
        "n_failed": len(failures),
        "runs_root": str(runs_root),
        "features_csv": str(feature_path),
    }
    (runs_root / "flow_topology_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
