import argparse
import json
from pathlib import Path

import pandas as pd


def main()  :
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Authoritative episode metadata CSV, joined one-to-one by episode_id")
    args = parser.parse_args()
    root = args.runs_root.expanduser().resolve()
    flow_path = root / "optical_flow_bout_features.csv"
    topology_path = root / "flow_topology_bout_features.csv"
    for path in (flow_path, topology_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    flow = pd.read_csv(flow_path)
    topology = pd.read_csv(topology_path)
    if flow["episode_id"].duplicated().any() or topology["episode_id"].duplicated().any():
        raise ValueError("Feature inputs contain duplicate episode_id values")
    shared = [c for c in flow.columns if c in topology.columns and c != "episode_id"]
    merged = flow.merge(topology.drop(columns=shared), on="episode_id", how="outer", validate="one_to_one")
    if args.metadata:
        metadata = pd.read_csv(args.metadata, dtype={"episode_id":str})
        if metadata.episode_id.duplicated().any():raise ValueError('Duplicate metadata episode_id')
        columns = [c for c in metadata.columns if c != 'episode_id']
        merged = merged.drop(columns=[c for c in columns if c in merged]).merge(metadata,on='episode_id',how='left',validate='one_to_one')
    if 'start_sec' not in merged and 'core_start_frame' in merged:
        merged['start_sec'] = pd.to_numeric(merged.core_start_frame) / pd.to_numeric(merged.fps)
    if 'duration_sec' not in merged:
        ending = 'core_end_frame' if 'core_end_frame' in merged else 'core_end_frame_exclusive'
        merged['duration_sec'] = (pd.to_numeric(merged[ending])-pd.to_numeric(merged.core_start_frame))/pd.to_numeric(merged.fps)
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
    merged["prism_extent"] = merged["mask_frame_rms_orbit_radius"]
    merged["prism_periodicity"] = merged["flow_normalized_periodicity_score"]
    merged["prism_frequency_hz"] = merged["topo_flow_frequency_hz"]
    merged["prism_period_sec"] = merged["topo_flow_period_sec"]
    from prism.analysis.real_video import add_interbout_intervals
    merged = add_interbout_intervals(merged)

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    output = args.output.expanduser().resolve()
    merged.to_csv(output, index=False)
    report = {
        "n_bouts": len(merged),
        "intensity_available": int(merged["prism_intensity"].notna().sum()),
        "extent_available": int(merged["prism_extent"].notna().sum()),
        "periodicity_available": int(merged["prism_periodicity"].notna().sum()),
        "frequency_available": int(merged["prism_frequency_hz"].notna().sum()),
        "output": str(output),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
