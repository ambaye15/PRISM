import numpy as np

from prism.optical_flow import (
    OpticalFlowSettings,
    analyze_optical_flow_arrays,
)
from prism.topology import (
    TopologySettings,
    analyze_flow_states,
    analyze_masks,
    framewise_rms_extent,
)


def optical_flow_intensity(
    frames ,
    masks ,
    fps ,
    flow_output=None,
)         :

    timestamps = np.arange(len(frames), dtype=np.float64) / float(fps)
    settings = OpticalFlowSettings(
        camera_correction="none",
        torso_correction=True,
        minimum_valid_pixels=25,
        torso_minimum_pixels=12,
        maximum_flow_dimension=768,
    )
    if flow_output is None:
        flow_output = analyze_optical_flow_arrays(
            frames,
            masks,
            timestamps,
            settings=settings,
        )
    result = flow_output["measurements"]
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
    from prism.topology.core import resolve_period
    from prism.topology.masks import longest_valid_run
    run = longest_valid_run(masks.reshape(len(masks), -1).any(axis=1))
    period, _ = resolve_period(masks[run], timestamps[run], settings)
    return {"period": period}


def topology_measurements(
    frames,
    masks ,
    fps ,
    dimension  = 23,
    n_points  = 300,
    flow_output=None,
)         :

    timestamps = np.arange(len(masks), dtype=np.float64) / float(fps)
    flow_settings = OpticalFlowSettings(
        camera_correction="none",
        torso_correction=True,
        minimum_valid_pixels=25,
        torso_minimum_pixels=12,
        maximum_flow_dimension=768,
    )
    if flow_output is None:
        flow_output = analyze_optical_flow_arrays(
            frames, masks, timestamps, settings=flow_settings
        )
    result = {}
    errors = []
    try:
        extent = framewise_rms_extent(masks)
        result["prism_extent"] = extent["extent_frame_rms"]
    except (ValueError, RuntimeError) as error:
        errors.append("extent: " + str(error))
    try:
        states = flow_output["flow_states"]
        settings = TopologySettings(d=dimension, n_points=n_points,
                                    frequency_method="lagged_ssm")
        flow = analyze_flow_states(states["total"], states["confidence"],
                                  states["timestamps_sec"],
                                  pair_indices=states["pair_indices"], settings=settings)
        period = flow["summary"]["period"]
        result.update({"prism_frequency_hz": period["frequency_hz"],
                       "prism_period_sec": period["period_sec"],
                       "flow_spectral_concentration": period.get("spectral_concentration"),
                       "prism_periodicity": flow["summary"]["normalized"].get("periodicity_score")})
        errors.extend("periodicity: " + item["error"] for item in flow["summary"].get("issues", []))
    except (ValueError, RuntimeError) as error:
        errors.append("flow_frequency: " + str(error))
    for method, column in [("lagged_ssm", "prism_frequency_lagged_ssm_hz"),
                           ("welch_pc", "prism_frequency_welch_hz")]:
        try:
            baseline = _topology_summary(masks, fps, method=method,
                                         dimension=dimension, n_points=n_points)
            result[column] = baseline["period"]["frequency_hz"]
        except (ValueError, RuntimeError) as error:
            errors.append(method + ": " + str(error))
    result["analysis_error"] = "; ".join(errors)
    for column in ["prism_extent", "prism_periodicity", "prism_frequency_hz",
                   "prism_frequency_lagged_ssm_hz", "prism_frequency_welch_hz"]:
        result[column + "_available"] = result.get(column) is not None and np.isfinite(result[column])
    return result


def production_flow_output(frames, masks, fps):

    timestamps = np.arange(len(frames), dtype=np.float64) / float(fps)
    settings = OpticalFlowSettings(
        camera_correction="none",
        torso_correction=True,
        minimum_valid_pixels=25,
        torso_minimum_pixels=12,
        maximum_flow_dimension=768,
    )
    return analyze_optical_flow_arrays(
        frames, masks, timestamps, settings=settings
    )
