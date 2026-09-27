import numpy as np

from prism.optical_flow import OpticalFlowSettings, summarize_flow_pair


def test_torso_centering_removes_bulk_translation_but_retains_limb_motion():
    mask = np.zeros((80, 60), dtype=bool)
    mask[10:70, 15:45] = True
    flow = np.zeros((80, 60, 2), dtype=np.float32)
    flow[..., 0][mask] = 2.0
    # Extra motion in the lateral limb regions, outside the central torso core.
    flow[20:50, 15:20, 0] += 4.0
    flow[20:50, 40:45, 0] -= 4.0
    settings = OpticalFlowSettings(
        camera_correction="none",
        region_mode="source",
        forward_backward_check=False,
        minimum_valid_pixels=10,
        minimum_retained_pixel_fraction=0.1,
        torso_minimum_pixels=10,
    )
    result = summarize_flow_pair(
        flow,
        mask,
        mask,
        delta_time=0.1,
        settings=settings,
    )
    assert result["flow_pair_valid"]
    assert result["flow_torso_correction_valid"]
    assert result["flow_torso_translation_body_heights_per_sec"] > 0
    assert (
        result["flow_pixel_p75_torso_corrected_body_heights_per_sec"]
        < result["flow_pixel_p75_body_heights_per_sec"]
    )
