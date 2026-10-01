import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform


def fill_short_gaps(values):
    values = np.asarray(values, dtype=float).copy()
    time = np.arange(len(values))
    for joint in range(values.shape[1]):
        for coordinate in range(values.shape[2]):
            series = values[:, joint, coordinate]
            valid = np.isfinite(series)
            if valid.sum() < 2:
                continue
            series[~valid] = np.interp(time[~valid], time[valid], series[valid])
            values[:, joint, coordinate] = series
    return values


def body_height(points):
    points = np.asarray(points, dtype=float)
    spans = np.nanmax(points, axis=1) - np.nanmin(points, axis=1)
    values = np.linalg.norm(spans, axis=1)
    return float(np.nanmedian(values))


def center_trajectory(points, torso_indices=None):
    points = fill_short_gaps(points)
    if torso_indices is None:
        center = np.nanmedian(points, axis=1, keepdims=True)
    else:
        center = np.nanmean(points[:, torso_indices], axis=1, keepdims=True)
    return points - center


def intensity(points, fps, torso_indices=None):
    height = body_height(points)
    centered = center_trajectory(points, torso_indices)
    speed = np.linalg.norm(np.diff(centered, axis=0), axis=2) * fps / height
    return float(np.nanmedian(np.nanquantile(speed, 0.75, axis=1)))


def estimate_frequency(points, fps, minimum_hz=0.2, maximum_hz=5.0):
    centered = center_trajectory(points)
    states = centered.reshape(len(centered), -1)
    distance = squareform(pdist(states))
    max_lag = min(len(states) - 2, int(np.floor(fps / minimum_hz)))
    min_lag = max(1, int(np.ceil(fps / maximum_hz)))
    if max_lag <= min_lag:
        return np.nan
    lags = np.arange(min_lag, max_lag + 1)
    curve = np.asarray([
        np.nanmedian(np.diag(distance, k=int(lag))) for lag in lags
    ])
    smooth = gaussian_filter1d(curve, sigma=0.75, mode="nearest")
    scale = float(np.nanmax(smooth) - np.nanmin(smooth))
    peaks, properties = find_peaks(
        -smooth,
        prominence=max(0.05 * scale, 1e-12),
    )
    if not len(peaks):
        index = int(np.nanargmin(smooth))
    else:
        # Recurrence minima also occur at integer multiples of the period.
        # Select the earliest peak whose prominence is at least one quarter of
        # the strongest recurrence peak, rather than selecting the deepest
        # (often a late harmonic) minimum.
        strong = peaks[
            properties["prominences"] >= 0.25 * properties["prominences"].max()
        ]
        index = int(strong[0])
    # Quadratic interpolation removes integer-lag frequency quantization.
    refined = float(lags[index])
    if 0 < index < len(curve) - 1:
        left, middle, right = smooth[index - 1:index + 2]
        denominator = left - 2.0 * middle + right
        if abs(denominator) > 1e-12:
            refined += float(0.5 * (left - right) / denominator)
    return float(fps / refined)


def sliding_window(points, fps, frequency_hz, dimension=23, n_points=200):
    centered = center_trajectory(points).reshape(len(points), -1)
    period_samples = fps / frequency_hz
    delay = max(period_samples / dimension, 1.0)
    starts = np.linspace(
        0.0,
        len(centered) - 1.0 - dimension * delay,
        min(n_points, max(8, len(centered) - dimension)),
    )
    if len(starts) < 8 or starts[-1] <= 0:
        raise ValueError("sequence is too short for a sliding-window orbit")
    sample_times = starts[:, None] + delay * np.arange(dimension + 1)[None]
    time = np.arange(len(centered), dtype=float)
    coordinates = []
    for feature in range(centered.shape[1]):
        coordinates.append(np.interp(sample_times.ravel(), time, centered[:, feature]))
    cloud = np.stack(coordinates, axis=1).reshape(len(starts), -1)
    cloud -= cloud.mean(axis=0, keepdims=True)
    return cloud


def extent_and_periodicity(points, fps, frequency_hz, dimension=23, n_points=200):
    from ripser import ripser

    height = body_height(points)
    cloud = sliding_window(points, fps, frequency_hz, dimension, n_points) / height
    radius = float(np.sqrt(np.mean(np.sum(cloud * cloud, axis=1))))
    norms = np.linalg.norm(cloud, axis=1)
    valid = norms > 1e-10
    normalized = cloud[valid] / norms[valid, None]
    normalized_distance = squareform(pdist(normalized))
    diagram = ripser(
        normalized_distance,
        distance_matrix=True,
        maxdim=1,
    )["dgms"][1]
    finite = diagram[np.isfinite(diagram[:, 1])]
    if not len(finite):
        score = 0.0
    else:
        persistence = finite[:, 1] - finite[:, 0]
        order = np.argsort(persistence)[::-1]
        first = finite[order[0]]
        first_term = np.exp(-first[0] / 2.0) * persistence[order[0]] / np.sqrt(3.0)
        second_term = 0.0
        if len(order) > 1:
            second = finite[order[1]]
            second_term = np.exp(-second[0] / 2.0) * persistence[order[1]] / np.sqrt(3.0)
        score = float(first_term - second_term)
    return radius, score


def all_measurements(points, fps, torso_indices=None):
    points = fill_short_gaps(points)
    frequency = estimate_frequency(points, fps)
    result = {
        "intensity": intensity(points, fps, torso_indices),
        "frequency_hz": frequency,
        "extent": np.nan,
        "periodicity": np.nan,
    }
    if np.isfinite(frequency):
        try:
            result["extent"], result["periodicity"] = extent_and_periodicity(
                points, fps, frequency
            )
        except (ValueError, RuntimeError):
            pass
    return result

