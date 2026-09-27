#!/usr/bin/env python3
"""Merge finalized PRISM intensity, extent, and periodicity measurements."""


import argparse
import json
from pathlib import Path

import pandas as pd


def main()  :
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.runs_root.expanduser().resolve()
    flow_path = root / "optical_flow_bout_features.csv"
    topology_path = root / "sliding_window_topology_bout_features.csv"
    for path in (flow_path, topology_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    flow = pd.read_csv(flow_path)
    topology = pd.read_csv(topology_path)
    if flow["episode_id"].duplicated().any() or topology["episode_id"].duplicated().any():
        raise ValueError("Feature inputs contain duplicate episode_id values")
    shared = [c for c in flow.columns if c in topology.columns and c != "episode_id"]
    merged = flow.merge(topology.drop(columns=shared), on="episode_id", how="outer", validate="one_to_one")
    total_column = "optical_flow_pixel_p75_body_heights_per_sec_median"
    torso_column = "optical_flow_pixel_p75_torso_corrected_body_heights_per_sec_median"
    merged["prism_intensity_total"] = merged[total_column]
    if torso_column not in merged.columns:
        raise ValueError(
            "Torso-corrected optical-flow column is missing. Rerun "
            "prism.run_optical_flow with torso correction enabled."
        )
    merged["prism_intensity_torso_corrected"] = merged[torso_column]
    merged["prism_intensity"] = merged[torso_column]
    torso_speed = "optical_flow_torso_translation_body_heights_per_sec_median"
    merged["prism_torso_translation_speed"] = merged.get(torso_speed)
    merged["prism_extent"] = merged["fixed_scale_point_cloud_orbit_radius"]
    merged["prism_periodicity"] = merged["fixed_normalized_periodicity_score"]
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    output = args.output.expanduser().resolve()
    merged.to_csv(output, index=False)
    report = {
        "n_bouts": len(merged),
        "intensity_available": int(merged["prism_intensity"].notna().sum()),
        "extent_available": int(merged["prism_extent"].notna().sum()),
        "periodicity_available": int(merged["prism_periodicity"].notna().sum()),
        "output": str(output),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
