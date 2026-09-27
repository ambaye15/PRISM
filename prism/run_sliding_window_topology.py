"""Batch SW1PerS analysis for fixed-crop SAM2 masks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ._records import asdict
from .topology import TopologySettings, analyze_masks, save_analysis

SCHEMA = "prism_sliding_window_topology.v3"


def build_parser():
    parser = argparse.ArgumentParser(
        description = "Run fixed-crop scale and sphere-normalized SW1PerS analysis."
    )
    parser.add_argument("--runs-root", type = Path, default = Path("prism/runs_block_filtered"))
    parser.add_argument("--episode-id", action = "append", default = [])
    parser.add_argument("--limit", type = int)
    parser.add_argument("--period-sec", type = float)
    parser.add_argument("--d", type = int, default = 23)
    parser.add_argument("--n-points", type = int, default = 300)
    parser.add_argument("--min-period-sec", type = float, default = 0.40)
    parser.add_argument("--max-period-sec", type = float, default = 5.0)
    parser.add_argument("--min-cycles", type = float, default = 2.5)
    parser.add_argument("--spectral-concentration-min", type = float, default = 0.15)
    parser.add_argument(
        "--frequency-method",
        choices=("lagged_ssm", "welch_pc"),
        default="lagged_ssm",
    )
    parser.add_argument("--welch-components", type = int, default = 3)
    parser.add_argument("--density-k", type = int, default = 20)
    parser.add_argument("--density-retain-fraction", type = float, default = 0.80)
    parser.add_argument("--minimum-valid-mask-fraction", type = float, default = 0.80)
    parser.add_argument(
        "--periodicity-score-min",
        type=float,
        help="Optional study-specific PS threshold; omitted by default.",
    )
    parser.add_argument("--no-h1", action = "store_true")
    parser.add_argument("--no-figures", action = "store_true")
    parser.add_argument("--force", action = "store_true")
    parser.add_argument("--fail-fast", action = "store_true")
    return parser


def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding = "utf-8"))


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline = "", encoding = "utf-8") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames = keys)
        writer.writeheader()
        writer.writerows(rows)


def identifiers(source, episode_id):
    return {
        "episode_id": episode_id,
        "fps": source.get("fps"),
        **(source.get("episode") or {}),
    }


def topology_measurements(result, valid_fraction, settings, score_min):
    summary = result["summary"]
    period = summary["period"]
    scale = summary["scale"]
    normalized = summary["normalized"]
    period_reliable = bool(
        period.get("reliable")
        and valid_fraction >= settings.minimum_valid_mask_fraction
    )
    score = normalized.get("periodicity_score")
    topological_reliable = bool(
        period_reliable
        and score is not None
        and (score_min is None or float(score) >= score_min)
    )
    cycle_distance = normalized.get("cycle_return_distance_median")
    return {
        "topo_mask_period_sec": period.get("period_sec"),
        "topo_mask_frequency_hz": period.get("frequency_hz"),
        "topo_mask_spectral_concentration": period.get("spectral_concentration"),
        "topo_mask_estimated_cycles": period.get("estimated_cycles"),
        "topo_mask_period_reliable": period_reliable,
        "topo_mask_temporal_measurement_reliable": topological_reliable,
        "topo_mask_valid_fraction": valid_fraction,
        "topo_mask_embedding_dimension": settings.d,
        "topo_mask_window_samples": settings.d + 1,
        "topo_mask_delay_sec": summary["embedding"]["tau_sec"],
        "topo_mask_window_span_sec": summary["embedding"]["window_span_sec"],
        "topo_mask_n_delay_points": summary["embedding"]["n_points_retained"],
        "topo_mask_n_delay_points_before_filter": summary["embedding"]["n_points_before_filter"],
        "topo_mask_density_retain_fraction": settings.density_retain_fraction,
        "fixed_scale_h1_computed": bool(scale.get("h1_computed", False)),
        "fixed_scale_h1_max_persistence": scale.get("h1_max_persistence"),
        "fixed_scale_h1_total_persistence": scale.get("h1_total_persistence"),
        "fixed_scale_h1_relative_persistence": scale.get("h1_relative_persistence"),
        "fixed_scale_point_cloud_orbit_radius": scale.get("point_cloud_orbit_radius"),
        "fixed_normalized_h1_computed": bool(normalized.get("h1_computed", False)),
        "fixed_normalized_h1_max_persistence": normalized.get("h1_max_persistence"),
        "fixed_normalized_h1_total_persistence": normalized.get("h1_total_persistence"),
        "fixed_normalized_h1_relative_persistence": normalized.get("h1_relative_persistence"),
        "fixed_normalized_periodicity_score": score,
        "fixed_normalized_point_cloud_orbit_radius": normalized.get("point_cloud_orbit_radius"),
        "fixed_normalized_cycle_return_distance": cycle_distance,
        "fixed_cycle_consistency": (
            None
            if cycle_distance is None
            else float(np.clip(1.0 - cycle_distance / 2.0, 0.0, 1.0))
        ),
        "centered_normalized_h1_computed": False,
        "centered_normalized_periodicity_score": None,
        "centered_scale_h1_computed": False,
        "centered_scale_h1_max_persistence": None,
        "centered_scale_point_cloud_orbit_radius": None,
        "centered_cycle_consistency": None,
    }


def analyze_episode(episode_dir, settings, score_min, create_figure = True):
    mask_path = episode_dir / "sam2_masks_fixed_crop.npz"
    with np.load(mask_path, allow_pickle = False) as saved:
        masks = saved["masks"].astype(bool)
        timestamps = saved["timestamps_sec"].astype(np.float64)
        core = saved["core_mask"].astype(bool)
        observed = masks.reshape(len(masks), -1).any(axis = 1)
        valid = (
            saved["mask_valid"].astype(bool)
            if "mask_valid" in saved.files
            else observed
        )
    core_count = int(np.count_nonzero(core))
    valid_fraction = (
        float(np.count_nonzero(valid & core) / core_count)
        if core_count else 0.0
    )
    result = analyze_masks(
        masks,
        timestamps,
        valid_mask = valid & core,
        settings = settings,
    )
    output = episode_dir / "sliding_window_topology"
    files = save_analysis(
        result,
        output,
        input_metadata = {
            "source": str(mask_path),
            "representation": "fixed body-scale crop",
            "core_valid_mask_fraction": valid_fraction,
        },
        title = episode_dir.name,
        create_figure = create_figure,
    )
    measurements = topology_measurements(
        result,
        valid_fraction,
        settings,
        score_min,
    )
    payload = {
        "schema_version": SCHEMA,
        "episode_id": episode_dir.name,
        "settings": {
            **asdict(settings),
            "minimum_valid_mask_fraction": settings.minimum_valid_mask_fraction,
            "periodicity_score_min": score_min,
        },
        "measurements": measurements,
        "files": files,
    }
    (episode_dir / "sliding_window_topology_summary.json").write_text(
        json.dumps(payload, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8",
    )
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
        d = args.d,
        n_points = args.n_points,
        period_sec = args.period_sec,
        min_period_sec = args.min_period_sec,
        max_period_sec = args.max_period_sec,
        min_cycles = args.min_cycles,
        spectral_concentration_min = args.spectral_concentration_min,
        frequency_method = args.frequency_method,
        welch_components = args.welch_components,
        density_k = args.density_k,
        density_retain_fraction = args.density_retain_fraction,
        compute_h1 = not args.no_h1,
    )
    settings.minimum_valid_mask_fraction = args.minimum_valid_mask_fraction
    expected_settings = {
        **asdict(settings),
        "minimum_valid_mask_fraction": args.minimum_valid_mask_fraction,
        "periodicity_score_min": args.periodicity_score_min,
    }
    topology_rows = []
    combined_rows = []
    failures = []
    completed = 0
    reused = 0
    for index, episode_dir in enumerate(episode_dirs, start = 1):
        print("[%d/%d] %s" % (index, len(episode_dirs), episode_dir.name))
        source = read_json(episode_dir / "subject_dino_summary.json")
        recurrence = read_json(episode_dir / "mask_recurrence_summary.json")
        summary_path = episode_dir / "sliding_window_topology_summary.json"
        try:
            existing = read_json(summary_path)
            figure_exists = (
                episode_dir / "sliding_window_topology" / "topology_diagnostic.png"
            ).is_file()
            if (
                not args.force
                and existing.get("schema_version") == SCHEMA
                and existing.get("settings") == expected_settings
                and (args.no_figures or figure_exists)
            ):
                topology = existing
                reused += 1
            else:
                topology = analyze_episode(
                    episode_dir,
                    settings,
                    args.periodicity_score_min,
                    create_figure = not args.no_figures,
                )
                completed += 1
            ids = identifiers(source, episode_dir.name)
            measurements = topology.get("measurements") or {}
            topology_rows.append({**ids, **measurements})
            combined_rows.append({
                **ids,
                **(source.get("measurements") or {}),
                **(recurrence.get("measurements") or {}),
                **measurements,
            })
        except Exception as error:
            failures.append({
                "episode_id": episode_dir.name,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            print("  FAILED: %s: %s" % (type(error).__name__, error))
            if args.fail_fast:
                raise

    topology_path = runs_root / "sliding_window_topology_bout_features.csv"
    combined_path = runs_root / "subject_dino_mask_topology_bout_features.csv"
    write_csv(topology_path, topology_rows)
    write_csv(combined_path, combined_rows)
    write_csv(
        runs_root / "subject_dino_topological_mask_bout_features.csv",
        combined_rows,
    )
    write_csv(runs_root / "sliding_window_topology_failures.csv", failures)
    report = {
        "schema_version": "prism_sliding_window_topology_batch.v1",
        "n_selected": len(episode_dirs),
        "n_completed": completed,
        "n_reused": reused,
        "n_failed": len(failures),
        "runs_root": str(runs_root),
        "features_csv": str(combined_path),
    }
    (runs_root / "sliding_window_topology_report.json").write_text(
        json.dumps(report, indent = 2) + "\n",
        encoding = "utf-8",
    )
    print(json.dumps(report, indent = 2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
