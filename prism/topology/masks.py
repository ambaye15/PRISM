import numpy as np

from .._records import field, record


@record
class TopologySettings:
    """Parameters for one mask sequence.

    ``d=23`` gives 24 mask samples per sliding-window point. If ``period_sec``
    is supplied it is used directly. Otherwise PRISM uses the lag profile of
    the mask self-distance matrix. PC-score Welch is retained as an ablation.
    """

    d = field(23)
    n_points = field(300)
    period_sec = field(None)
    min_period_sec = field(0.4)
    max_period_sec = field(5.0)
    min_cycles = field(2.5)
    spectral_concentration_min = field(0.15)
    frequency_method = field("lagged_ssm")
    welch_components = field(3)
    welch_nyquist_fraction = field(0.40)
    density_k = field(20)
    density_retain_fraction = field(0.80)
    compute_h1 = field(True)

def validate_sequence(masks, timestamps):
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) < 8:
        raise ValueError("masks must have shape (frames, height, width) with >=8 frames")
    masks = masks.astype(bool)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) != len(masks):
        raise ValueError("timestamps and masks must contain the same number of frames")
    if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return masks, timestamps

def require_nonempty_masks(masks):
    empty = np.flatnonzero(~masks.reshape(len(masks), -1).any(axis=1))
    if len(empty):
        preview = ", ".join(str(int(index)) for index in empty[:10])
        raise ValueError(f"analyzed masks must be nonempty; empty indices: {preview}")

def longest_valid_run(valid, minimum = 8):
    valid = np.asarray(valid, dtype=bool)
    best_start = best_stop = 0
    start = None
    for index in range(len(valid) + 1):
        active = index < len(valid) and bool(valid[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start > best_stop - best_start:
                best_start, best_stop = start, index
            start = None
    if best_stop - best_start < minimum:
        raise ValueError(f"no uninterrupted valid run contains at least {minimum} frames")
    return np.arange(best_start, best_stop, dtype=int)

def mask_geometry(masks):
    masks = np.asarray(masks, dtype=bool)
    flat = masks.reshape(len(masks), -1).astype(np.float32)
    areas = flat.sum(axis=1, dtype=np.float64)
    gram = np.asarray(flat @ flat.T, dtype=np.float64)
    squared = np.maximum(areas[:, None] + areas[None, :] - 2.0 * gram, 0.0)
    median_area = float(np.median(areas))
    body_distance = np.sqrt(squared / max(median_area, 1.0))
    unions = areas[:, None] + areas[None, :] - gram
    jaccard = np.zeros_like(unions)
    nonzero = unions > 0
    jaccard[nonzero] = 1.0 - gram[nonzero] / unions[nonzero]
    np.fill_diagonal(jaccard, 0.0)
    return {
        "gram": gram,
        "areas": areas,
        "median_area": median_area,
        "body_distance": body_distance,
        "jaccard_distance": np.clip(jaccard, 0.0, 1.0),
    }
