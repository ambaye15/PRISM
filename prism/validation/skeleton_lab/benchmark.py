"""Generate and run a multi-behavior PRISM nuisance benchmark."""


import json
from pathlib import Path

import numpy as np
import pandas as pd
from skeleton_lab.controls import set_by_path
from skeleton_lab.export import export
from skeleton_lab.simulate import simulate
from skeleton_lab.spec import ScenarioSpec

from .analysis import extract_truth
from .degradation import degrade_sequence
from .masks import render_exact_sequence
from .measurements import optical_flow_intensity, topology_measurements
from .stabilization import estimate_background_warps, stabilize_sequence


class Behavior:
    """Configuration for one synthetic movement geometry."""

    __slots__ = (
        "name",
        "preset",
        "drive",
        "frequency_path",
        "amplitude_path",
        "extent_dof",
    )

    def __init__(self, name, preset, drive, frequency_path, amplitude_path, extent_dof):
        self.name = name
        self.preset = preset
        self.drive = drive
        self.frequency_path = frequency_path
        self.amplitude_path = amplitude_path
        self.extent_dof = extent_dof


BEHAVIORS = (
    Behavior("arm_flapping", "arm_flapping.json", "flap",
             "drives[flap].frequency.value", "drives[flap].amplitude.tracks[1].value",
             "shoulder_l.abduction"),
    Behavior("hand_flapping", "hand_flapping.json", "flap",
             "drives[flap].frequency.value", "drives[flap].amplitude.tracks[1].value",
             "wrist_l.flexion"),
    Behavior("body_rocking", "body_rocking.json", "rock",
             "shared_tracks[rock_f].value", "drives[rock].amplitude.tracks[1].value",
             "spine.pitch"),
    Behavior("head_banging", "head_banging.json", "bang",
             "drives[bang].frequency.value", "drives[bang].amplitude.tracks[1].value",
             "neck.pitch"),
    Behavior("head_shaking", "head_shaking.json", "shake",
             "drives[shake].frequency.value", "drives[shake].amplitude.tracks[1].value",
             "neck.yaw"),
)

DEGRADATION_DEFAULTS   = {
    "fps_keep_fraction": 1.0,
    "spatial_scale": 1.0,
    "gaussian_blur_sigma": 0.0,
    "pixel_noise_sigma": 0.0,
    "jpeg_quality": 100.0,
    "mask_boundary_jitter_px": 0.0,
    "mask_flip_probability": 0.0,
    "mask_missing_fraction": 0.0,
    "mask_missing_block_fraction": 0.0,
    "mask_occlusion_fraction": 0.0,
    "mask_morphology_px": 0.0,
}


def _payload(behavior , frequency , duration , amplitude_scale ,
             seed , specs_root )   :
    payload = json.loads((specs_root / behavior.preset).read_text())
    base_amplitude = payload["drives"][next(
        i for i, drive in enumerate(payload["drives"]) if drive["id"] == behavior.drive
    )]["amplitude"]["tracks"][1]["value"]
    set_by_path(payload, behavior.frequency_path, frequency)
    set_by_path(payload, behavior.amplitude_path, float(base_amplitude) * amplitude_scale)
    payload["duration_sec"] = duration
    payload["seed"] = seed
    payload["name"] = (
        f"{behavior.name}_f{frequency:g}_T{duration:g}_A{amplitude_scale:g}_seed{seed}"
    )
    # Clean physical motion: nuisance is added in explicitly labeled benchmark conditions.
    payload["drives"][0]["amplitude"]["value"] = 0.0
    payload["nuisance"]["sway"]["enabled"] = False
    payload["nuisance"]["micro_motion"]["enabled"] = False
    return payload


def generate_benchmark(
    output_root ,
    specs_root  = Path("examples/specs"),
    frequencies   = (0.75, 1.0, 2.0, 3.0, 4.0),
    durations   = (2.0, 3.0, 5.0, 7.5, 10.0),
    amplitude_scales   = (0.6, 1.0, 1.4),
    seeds   = (0, 1, 2),
)  :
    """Create truth-only bundles and a matched analysis-condition manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    bundles_root = output_root / "bundles"
    rows   = []

    # Clean factorial cells establish construct validity and duration breakdown.
    for behavior in BEHAVIORS:
        for frequency in frequencies:
            for duration in durations:
                for amplitude_scale in amplitude_scales:
                    for seed in seeds:
                        payload = _payload(behavior, frequency, duration, amplitude_scale, seed, specs_root)
                        bundle = bundles_root / payload["name"]
                        if not (bundle / "truth.json").exists():
                            export(simulate(ScenarioSpec.model_validate(payload)), bundle, write_video=False)
                        rows.append({
                            "condition_id": payload["name"] + "__clean",
                            "bundle_path": str(bundle), "behavior": behavior.name,
                            "drive": behavior.drive, "extent_dof": behavior.extent_dof,
                            "family": "construct", "nuisance": "clean", "level": 0.0,
                            "frequency_truth_hz": frequency, "duration_sec": duration,
                            "amplitude_scale": amplitude_scale, "seed": seed,
                            "match_id": payload["name"],
                        })

    # One-factor perturbations use a fixed, well-estimable eight-second motion.
    nuisance_ladders = {
        "fps_keep_fraction": (0.25, 0.5, 0.75),
        "spatial_scale": (0.25, 0.5, 0.75),
        "gaussian_blur_sigma": (0.75, 1.5, 3.0),
        "pixel_noise_sigma": (5.0, 15.0, 30.0),
        "jpeg_quality": (75, 50, 25),
        "mask_boundary_jitter_px": (0.5, 1.5, 3.0),
        "mask_flip_probability": (0.01, 0.05, 0.10),
        "mask_missing_fraction": (0.05, 0.10, 0.20),
        "mask_missing_block_fraction": (0.05, 0.10, 0.20),
        "mask_occlusion_fraction": (0.05, 0.10, 0.20),
        "mask_morphology_px": (-2, -1, 1, 2),
    }
    for behavior in BEHAVIORS:
        for seed in seeds:
            payload = _payload(behavior, 2.0, 8.0, 1.0, seed, specs_root)
            bundle = bundles_root / payload["name"]
            if not (bundle / "truth.json").exists():
                export(simulate(ScenarioSpec.model_validate(payload)), bundle, write_video=False)
            match_id = f"{behavior.name}__robustness_seed{seed}"
            base = {"bundle_path": str(bundle), "behavior": behavior.name,
                    "drive": behavior.drive, "extent_dof": behavior.extent_dof,
                    "frequency_truth_hz": 2.0, "duration_sec": 8.0,
                    "amplitude_scale": 1.0, "seed": seed, "match_id": match_id}
            rows.append({**base, "condition_id": payload["name"] + "__baseline",
                         "family": "robustness", "nuisance": "clean", "level": 0.0})
            for nuisance, levels in nuisance_ladders.items():
                for level in levels:
                    rows.append({**base,
                        "condition_id": payload["name"] + f"__{nuisance}_{level}",
                        "family": "robustness", "nuisance": nuisance, "level": level,
                        nuisance: level,
                    })

            # Simulator-side camera perturbations need new bundles because they
            # alter the actual projection, rather than post-processing pixels.
            camera_ladders = {
                "camera_azimuth_deg": (30.0, 60.0, 90.0),
                "camera_elevation_deg": (-15.0, 20.0, 40.0),
                "camera_scale_factor": (0.60, 0.80, 1.25, 1.50),
                "camera_shake_sigma_px": (2.0, 5.0, 10.0),
                "camera_perspective": (0.10, 0.25, 0.40),
            }
            for nuisance, levels in camera_ladders.items():
                for level in levels:
                    camera_payload = _payload(behavior, 2.0, 8.0, 1.0, seed, specs_root)
                    if nuisance == "camera_azimuth_deg":
                        camera_payload["camera"]["azimuth_deg"] = level
                    elif nuisance == "camera_elevation_deg":
                        camera_payload["camera"]["elevation_deg"] = level
                    elif nuisance == "camera_scale_factor":
                        camera_payload["camera"]["px_per_m"] *= level
                    elif nuisance == "camera_shake_sigma_px":
                        camera_payload["camera"]["shake"]["sigma_px"] = level
                    elif nuisance == "camera_perspective":
                        camera_payload["camera"]["perspective"] = level
                    camera_payload["name"] += f"__{nuisance}_{level:g}"
                    camera_bundle = bundles_root / camera_payload["name"]
                    if not (camera_bundle / "truth.json").exists():
                        export(simulate(ScenarioSpec.model_validate(camera_payload)),
                               camera_bundle, write_video=False)
                    rows.append({**base, "bundle_path": str(camera_bundle),
                        "condition_id": camera_payload["name"],
                        "family": "robustness", "nuisance": nuisance, "level": level,
                    })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "benchmark_manifest.csv", index=False)
    return manifest


def run_benchmark(manifest_path , output_csv , max_dimension  = 256,
                  dimension  = 23, n_points  = 200,
                  only_family    = None, resume  = True,
                  camera_correction  = "none",
                  only_nuisances     = None)  :
    manifest = pd.read_csv(manifest_path)
    if only_family is not None:
        manifest = manifest[manifest["family"] == only_family].reset_index(drop=True)
    if only_nuisances:
        manifest = manifest[manifest["nuisance"].isin(only_nuisances)].reset_index(drop=True)
    rows   = []
    completed  = set()
    if resume and output_csv.exists():
        previous = pd.read_csv(output_csv)
        rows = previous.to_dict("records")
        completed = set(previous["condition_id"].astype(str))
        print(f"Resuming after {len(completed)} completed conditions", flush=True)
    for index, condition in manifest.iterrows():
        if str(condition.condition_id) in completed:
            continue
        print(f"[{index + 1}/{len(manifest)}] {condition.condition_id}", flush=True)
        bundle = Path(condition.bundle_path)
        frames, masks, fps = render_exact_sequence(
            bundle, max_dimension=max_dimension,
            remove_camera_motion=(camera_correction == "synthetic"),
        )
        kwargs = dict(DEGRADATION_DEFAULTS)
        for field in DEGRADATION_DEFAULTS:
            if field in manifest.columns and pd.notna(condition[field]):
                kwargs[field] = float(condition[field])
        kwargs["jpeg_quality"] = int(kwargs["jpeg_quality"])
        kwargs["mask_morphology_px"] = int(kwargs["mask_morphology_px"])
        frames, masks, retained = degrade_sequence(
            frames, masks, seed=int(condition.seed) + 1009, **kwargs
        )
        correction = {"camera_correction": camera_correction,
                      "camera_correction_valid_pair_fraction": np.nan,
                      "camera_correction_median_inliers": np.nan}
        if camera_correction == "synthetic":
            correction["camera_correction_valid_pair_fraction"] = 1.0
        elif camera_correction == "ransac":
            warps, diagnostic = estimate_background_warps(frames, masks)
            frames, masks = stabilize_sequence(frames, masks, warps)
            correction["camera_correction_valid_pair_fraction"] = diagnostic.valid_pair_fraction
            correction["camera_correction_median_inliers"] = diagnostic.median_inliers
        elif camera_correction != "none":
            raise ValueError("camera_correction must be none, synthetic, or ransac")
        effective_fps = fps / max(1, round(1.0 / max(float(kwargs.get("fps_keep_fraction", 1)), 1e-6)))
        behavior = next(item for item in BEHAVIORS if item.name == condition.behavior)
        row = condition.to_dict()
        row.update(correction)
        row.update(extract_truth(bundle, behavior.drive, behavior.extent_dof))
        row.update(optical_flow_intensity(frames, masks, effective_fps))
        try:
            row.update(topology_measurements(masks, effective_fps, dimension, n_points))
            row["analysis_error"] = ""
        except (ValueError, RuntimeError) as error:
            row["analysis_error"] = str(error)
        row["effective_fps"] = effective_fps
        row["true_cycles"] = float(condition.frequency_truth_hz * condition.duration_sec)
        rows.append(row)
        # A run can take hours; every completed condition is a safe checkpoint.
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    result = pd.DataFrame(rows)
    return result
