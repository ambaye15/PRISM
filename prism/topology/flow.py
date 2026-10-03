"""Frequency and persistent periodicity from signed optical-flow states."""

import numpy as np

from .core import cycle_return_distance, persistence_summary
from .embedding import interpolation_operators, knn_density_selection
from .masks import TopologySettings
from .period import _spectral_peak


def body_grid(flow, region, mask, body_height, delta_time, grid_shape):
    """Map a signed flow field to a fixed body-centred coordinate grid."""
    import cv2

    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None, None
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    weight = region[y0:y1, x0:x1].astype(np.float32)
    values = flow[y0:y1, x0:x1].astype(np.float32)
    height, width = int(grid_shape[0]), int(grid_shape[1])
    resized_weight = cv2.resize(weight, (width, height), interpolation=cv2.INTER_AREA)
    weighted = np.stack(
        [
            cv2.resize(
                values[..., component] * weight,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            for component in range(2)
        ],
        axis=-1,
    )
    valid = resized_weight > 1e-3
    state = np.zeros((height, width, 2), dtype=np.float64)
    state[valid] = weighted[valid] / resized_weight[valid, None]
    state /= max(float(body_height) * float(delta_time), 1e-12)
    confidence = np.clip(resized_weight.astype(np.float64), 0.0, 1.0)
    return state, confidence


def _robust_clip(states, confidence, clip_mad):
    result = np.asarray(states, dtype=np.float64).copy()
    magnitude = np.linalg.norm(result, axis=-1)
    usable = magnitude[np.asarray(confidence) > 0.05]
    if not len(usable) or clip_mad is None:
        return result, None
    median = float(np.median(usable))
    mad = float(np.median(np.abs(usable - median)))
    scale = 1.4826 * mad
    limit = median + float(clip_mad) * max(scale, median * 0.05, 1e-8)
    factor = np.minimum(1.0, limit / np.maximum(magnitude, 1e-12))
    result *= factor[..., None]
    return result, limit


def _longest_consecutive_run(pair_indices, minimum=8):
    pair_indices = np.asarray(pair_indices, dtype=int)
    if not len(pair_indices):
        raise ValueError("no valid optical-flow states")
    starts = np.r_[0, np.flatnonzero(np.diff(pair_indices) != 1) + 1]
    stops = np.r_[starts[1:], len(pair_indices)]
    lengths = stops - starts
    selected = int(np.argmax(lengths))
    if lengths[selected] < int(minimum):
        raise ValueError(
            "no uninterrupted run contains at least %d valid flow states" % minimum
        )
    return np.arange(starts[selected], stops[selected], dtype=int)


def _features(states, confidence):
    weight = np.sqrt(np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0))
    values = np.asarray(states, dtype=np.float64) * weight[..., None]
    return values.reshape(len(values), -1)


def _geometry(features):
    gram = np.asarray(features @ features.T, dtype=np.float64)
    norms = np.diag(gram)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * gram, 0.0)
    scale = max(float(np.median(np.sum(features * features, axis=1))), 1e-12)
    distance = np.sqrt(squared / scale)
    np.fill_diagonal(distance, 0.0)
    return gram, distance


def _period_from_distance(distance, timestamps, settings):
    max_lag = len(distance) // 2
    if max_lag < 8:
        raise ValueError("flow trajectory is too short for lagged-SSM frequency")
    profile = np.asarray(
        [np.mean(np.diag(distance, k=lag)) for lag in range(1, max_lag + 1)],
        dtype=np.float64,
    )
    dt = float(np.median(np.diff(timestamps)))
    if settings.period_sec is not None:
        period = float(settings.period_sec)
        if period <= 0:
            raise ValueError("period_sec must be positive")
        duration = float(timestamps[-1] - timestamps[0] + dt)
        cycles = duration / period
        return {
            "frequency_hz": float(1.0 / period),
            "period_sec": period,
            "estimated_cycles": float(cycles),
            "spectral_concentration": None,
            "reliable": bool(cycles >= settings.min_cycles),
            "source": "user_supplied",
            "frequency_hz_grid": np.asarray([], dtype=np.float64),
            "power": np.asarray([], dtype=np.float64),
            "lag_profile": profile,
        }
    frequency, grid, power, concentration = _spectral_peak(
        profile, 1.0 / dt, settings
    )
    if frequency is None:
        raise ValueError("automatic flow frequency estimation failed")
    duration = float(timestamps[-1] - timestamps[0] + dt)
    cycles = duration * float(frequency)
    return {
        "frequency_hz": float(frequency),
        "period_sec": float(1.0 / frequency),
        "estimated_cycles": float(cycles),
        "spectral_concentration": concentration,
        "reliable": bool(
            cycles >= settings.min_cycles
            and concentration is not None
            and concentration >= settings.spectral_concentration_min
        ),
        "source": "total_signed_flow_lagged_ssm",
        "frequency_hz_grid": grid,
        "power": power,
        "lag_profile": profile,
    }


def _window_distances(gram, timestamps, period_sec, d, n_points):
    tau = float(period_sec) / (int(d) + 1)
    span = int(d) * tau
    latest = float(timestamps[-1] - span)
    if latest <= timestamps[0]:
        raise ValueError("flow sequence is too short for period-adapted windows")
    starts = np.linspace(float(timestamps[0]), latest, int(n_points))
    operators = interpolation_operators(timestamps, starts, int(d), tau)
    raw_dot = np.zeros((len(starts), len(starts)), dtype=np.float64)
    summed = np.zeros_like(operators[0])
    for operator in operators:
        raw_dot += operator @ gram @ operator.T
        summed += operator
    raw_dot = 0.5 * (raw_dot + raw_dot.T)
    raw_norms = np.diag(raw_dot)
    raw_squared = np.maximum(
        raw_norms[:, None] + raw_norms[None, :] - 2.0 * raw_dot, 0.0
    )
    scale = np.sqrt(raw_squared / max(int(d) + 1, 1))

    mean_dot = (summed @ gram @ summed.T) / (int(d) + 1)
    centered = 0.5 * ((raw_dot - mean_dot) + (raw_dot - mean_dot).T)
    centered_norms = np.maximum(np.diag(centered), 0.0)
    valid = centered_norms > 1e-12
    if int(valid.sum()) < 8:
        raise ValueError("fewer than 8 nonconstant flow windows")
    centered = centered[np.ix_(valid, valid)]
    starts = starts[valid]
    scale = scale[np.ix_(valid, valid)]
    centered_norms = centered_norms[valid]
    cosine = np.clip(
        centered / np.sqrt(np.outer(centered_norms, centered_norms)), -1.0, 1.0
    )
    normalized = np.sqrt(np.maximum(2.0 - 2.0 * cosine, 0.0))
    np.fill_diagonal(scale, 0.0)
    np.fill_diagonal(normalized, 0.0)
    return starts, scale, normalized


def analyze_flow_states(
    states,
    confidence,
    timestamps,
    *,
    pair_indices=None,
    settings=None,
    clip_mad=6.0,
):
    """Estimate frequency and H1 periodicity from total signed flow."""
    settings = settings or TopologySettings()
    states = np.asarray(states, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if states.ndim != 4 or states.shape[-1] != 2:
        raise ValueError("states must have shape (time, height, width, 2)")
    if confidence.shape != states.shape[:-1] or len(timestamps) != len(states):
        raise ValueError("confidence and timestamps must match states")
    if pair_indices is None:
        pair_indices = np.arange(len(states), dtype=int)
    keep_run = _longest_consecutive_run(pair_indices)
    states = states[keep_run]
    confidence = confidence[keep_run]
    timestamps = timestamps[keep_run]
    states, clip_limit = _robust_clip(states, confidence, clip_mad)

    features = _features(states, confidence)
    gram, state_distance = _geometry(features)
    period = _period_from_distance(state_distance, timestamps, settings)
    starts, scale, normalized = _window_distances(
        gram,
        timestamps,
        period["period_sec"],
        settings.d,
        settings.n_points,
    )
    density = knn_density_selection(
        scale,
        k=settings.density_k,
        retain_fraction=settings.density_retain_fraction,
    )
    keep = np.flatnonzero(density["keep_mask"])
    starts = starts[keep]
    scale = scale[np.ix_(keep, keep)]
    normalized = normalized[np.ix_(keep, keep)]
    scale_summary, scale_diagram = persistence_summary(
        scale, settings.compute_h1, normalized_cloud=False
    )
    normalized_summary, normalized_diagram = persistence_summary(
        normalized, settings.compute_h1, normalized_cloud=True
    )
    return {
        "summary": {
            "representation": "camera-corrected total signed optical flow",
            "period": {
                key: value
                for key, value in period.items()
                if key not in {"frequency_hz_grid", "power", "lag_profile"}
            },
            "embedding": {
                "d": settings.d,
                "window_samples": settings.d + 1,
                "n_states": int(len(states)),
                "n_windows_retained": int(len(keep)),
            },
            "normalized": normalized_summary,
            "scale": scale_summary,
            "clip_limit_body_heights_per_sec": clip_limit,
        },
        "state_distance": state_distance,
        "scale_distance": scale,
        "normalized_distance": normalized,
        "scale_diagram": scale_diagram,
        "normalized_diagram": normalized_diagram,
        "start_times_sec": starts,
        "frequency_hz_grid": period["frequency_hz_grid"],
        "frequency_power": period["power"],
        "lag_profile": period["lag_profile"],
        "cycle_return_distance": cycle_return_distance(
            normalized, starts, period["period_sec"]
        ),
    }
