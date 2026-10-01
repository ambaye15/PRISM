import numpy as np
import pandas as pd

from prism.validation.agreement import agreement_rows
from prism.validation.pose_baseline.measurements import measure_pose_bout


def test_behavior_specific_participant_count_uses_paired_subset():
    data = pd.DataFrame({
        "participant_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
        "behavior": ["flap", "rock", "flap", "rock", "rock", "rock"],
        "pose": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
        "flow": [1.1, 2.1, 1.9, 3.2, 4.1, 5.1],
    })
    rows = agreement_rows(data, "pose", "flow", "pose_vs_flow", 20, 8)
    flap = next(
        row for row in rows
        if row["analysis"] == "behavior_spearman" and row["behavior"] == "flap"
    )
    rock = next(
        row for row in rows
        if row["analysis"] == "behavior_spearman" and row["behavior"] == "rock"
    )
    assert flap["n"] == 2
    assert flap["n_participants"] == 2
    assert rock["n"] == 4
    assert rock["n_participants"] == 3


def test_pose_motion_decomposition_separates_rigid_translation():
    frames = 8
    joints = 17
    xy = np.zeros((frames, joints, 2), dtype=float)
    xy[:, :, 0] = np.arange(frames)[:, None] * 2.0
    xy[:, :, 1] = np.linspace(10.0, 90.0, joints)[None, :]
    confidence = np.ones((frames, joints), dtype=float)
    masks = np.ones((frames, 100, 50), dtype=bool)
    result = measure_pose_bout(xy, confidence, masks, fps=10.0)
    assert result["pose_total_motion_p75_mask_heights_per_sec"] > 0.0
    assert result["pose_torso_translation_speed_mask_heights_per_sec"] > 0.0
    assert result["pose_articulated_motion_p75_mask_heights_per_sec"] < 1e-12


def test_pose_camera_correction_removes_known_image_translation():
    frames = 8
    joints = 17
    xy = np.zeros((frames, joints, 2), dtype=float)
    xy[:, :, 0] = np.arange(frames)[:, None] * 2.0
    xy[:, :, 1] = np.linspace(10.0, 90.0, joints)[None, :]
    confidence = np.ones((frames, joints), dtype=float)
    masks = np.ones((frames, 100, 50), dtype=bool)
    transforms = [np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])] * (frames - 1)
    result = measure_pose_bout(
        xy,
        confidence,
        masks,
        fps=10.0,
        camera_transforms=transforms,
    )
    assert result["pose_total_motion_p75_mask_heights_per_sec"] < 1e-12
    assert result["pose_torso_translation_speed_mask_heights_per_sec"] < 1e-12
