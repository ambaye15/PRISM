"""Adapters from Skeleton Lab arrays to the production PRISM API."""


import numpy as np

from prism.optical_flow import (
    OpticalFlowSettings,
    analyze_optical_flow_arrays,
)
from prism.topology import TopologySettings, analyze_masks


def optical_flow_intensity(
    frames ,
    masks ,
    fps ,
)         :
    """Return the manuscript's torso-centered intensity and its ablations."""
    timestamps = np.arange(len(frames), dtype=np.float64) / float(fps)
    settings = OpticalFlowSettings(
        camera_correction="none",
        torso_correction=True,
        minimum_valid_pixels=25,
        torso_minimum_pixels=12,
        maximum_flow_dimension=768,
    )
    result = analyze_optical_flow_arrays(
        frames,
        masks,
        timestamps,
        settings=settings,
    )["measurements"]
    total = result.get(
        "optical_flow_pixel_p75_body_heights_per_sec_median"
    )
    torso = result.get(
        "optical_flow_pixel_p75_torso_corrected_body_heights_per_sec_median"
    )
    return {
        "prism_intensity": torso,
        "prism_intensity_torso_corrected": torso,
        "prism_intensity_total": total,
        "prism_torso_translation_speed": result.get(
            "optical_flow_torso_translation_body_heights_per_sec_median"
        ),
        "flow_valid_pairs": result.get("optical_flow_n_valid_pairs"),
        "flow_valid_pair_fraction": result.get(
            "optical_flow_valid_pair_fraction"
        ),
        "flow_measurement_reliable": result.get(
            "optical_flow_torso_measurement_reliable"
        ),
    }


def _topology_summary(
    masks ,
    fps ,
    *,
    method ,
    dimension ,
    n_points ,
)  :
    timestamps = np.arange(len(masks), dtype=np.float64) / float(fps)
    settings = TopologySettings(
        d=dimension,
        n_points=n_points,
        frequency_method=method,
    )
    return analyze_masks(masks, timestamps, settings=settings)["summary"]


def topology_measurements(
    masks ,
    fps ,
    dimension  = 23,
    n_points  = 300,
)         :
    """Return primary lagged-SSM measures plus the Welch frequency ablation."""
    primary = _topology_summary(
        masks,
        fps,
        method="lagged_ssm",
        dimension=dimension,
        n_points=n_points,
    )
    welch = _topology_summary(
        masks,
        fps,
        method="welch_pc",
        dimension=dimension,
        n_points=n_points,
    )
    return {
        "prism_extent": primary["scale"]["point_cloud_orbit_radius"],
        "prism_periodicity": primary["normalized"]["periodicity_score"],
        "prism_frequency_lagged_ssm_hz": primary["period"]["frequency_hz"],
        "prism_frequency_welch_hz": welch["period"]["frequency_hz"],
        "prism_period_sec": primary["period"]["period_sec"],
        "lagged_ssm_spectral_concentration": primary["period"].get(
            "spectral_concentration"
        ),
        "welch_spectral_concentration": welch["period"].get(
            "spectral_concentration"
        ),
        "welch_components": welch["period"].get("n_components_used"),
        "welch_explained_variance": welch["period"].get("pc_variance_ratio"),
    }

