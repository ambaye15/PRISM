from .measurements import measure_pose_bout


def run_pose_baseline(*args, **kwargs):

    from .analysis import run_pose_baseline as implementation
    return implementation(*args, **kwargs)


__all__ = ["run_pose_baseline", "measure_pose_bout"]
