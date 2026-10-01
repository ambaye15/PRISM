"""Mask-aligned pose baseline for real-video convergent validation."""

from .measurements import measure_pose_bout


def run_pose_baseline(*args, **kwargs):
    # Keep measurement-only imports usable without OpenCV or Ultralytics.
    from .analysis import run_pose_baseline as implementation
    return implementation(*args, **kwargs)


__all__ = ["run_pose_baseline", "measure_pose_bout"]
