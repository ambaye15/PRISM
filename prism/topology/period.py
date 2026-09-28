import numpy as np

from .masks import TopologySettings, mask_geometry, require_nonempty_masks, validate_sequence


def _spectral_peak(signal, fps, settings, *, nfft_min=2048):
    """Return an interpolated in-band Welch peak and its diagnostics."""
    try:
        from scipy.signal import welch
    except ImportError as error:
        raise ImportError("SciPy is required for automatic period estimation") from error

    signal = np.asarray(signal, dtype=np.float64)
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(max(8 * len(signal), 2)))))
    frequencies, power = welch(
        signal,
        fs=fps,
        window="hann",
        nperseg=len(signal),
        noverlap=0,
        nfft=nfft,
        detrend="linear",
        scaling="density",
    )
   
    minimum_frequency = 1.0 / settings.max_period_sec
    maximum_frequency = min(
        1.0 / settings.min_period_sec,
        0.45 * fps,
    )
    eligible = (
        (frequencies >= minimum_frequency)
        & (frequencies <= maximum_frequency)
        & np.isfinite(power)
    )
    if not np.any(eligible) or float(power[eligible].sum()) <= 1e-12:
        return None, frequencies, power, None

    candidates = np.flatnonzero(eligible)
    peak = int(candidates[np.argmax(power[eligible])])
    offset = 0.0
    if 0 < peak < len(power) - 1:
        left, center, right = power[peak - 1 : peak + 2]
        denominator = left - 2.0 * center + right
        if abs(float(denominator)) > 1e-15:
            offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    spacing = float(frequencies[1] - frequencies[0])
    frequency = float(frequencies[peak] + offset * spacing)
   
    native_half_width = max(1, int(np.ceil(nfft / max(len(signal), 1))))
    lo = max(0, peak - native_half_width)
    hi = min(len(power), peak + native_half_width + 1)
    concentration = float(power[lo:hi].sum() / power[eligible].sum())
    return frequency, frequencies, power, concentration


def period_from_lagged_ssm(masks, timestamps, settings):
    """Estimate frequency from the mean upper diagonals of mask self-distance.

    For lag ``L``, the signal is ``s[L] = mean_i D(i, i + L)``. A periodic
    mask trajectory produces a periodic lag profile without requiring any one
    image coordinate to behave sinusoidally.
    """
    geometry = mask_geometry(masks)
    distance = np.asarray(geometry["body_distance"], dtype=np.float64)
    max_lag = len(distance) // 2
    if max_lag < 8:
        return {
            "period_sec": None,
            "frequency_hz": None,
            "clarity": None,
            "spectral_concentration": None,
            "estimated_cycles": None,
            "reliable": False,
            "source": "automatic_lagged_ssm",
        }, np.empty(0, dtype=np.float64)

    profile = np.asarray(
        [np.mean(np.diag(distance, k=lag)) for lag in range(1, max_lag + 1)],
        dtype=np.float64,
    )
    typical_dt = float(np.median(np.diff(timestamps)))
    fps = 1.0 / typical_dt
    frequency, grid, power, concentration = _spectral_peak(profile, fps, settings)
    if frequency is None:
        return {
            "period_sec": None,
            "frequency_hz": None,
            "clarity": None,
            "spectral_concentration": None,
            "estimated_cycles": None,
            "reliable": False,
            "source": "automatic_lagged_ssm",
            "frequency_hz_grid": grid,
            "welch_power": power,
        }, profile

    duration = float(timestamps[-1] - timestamps[0] + typical_dt)
    cycles = duration * frequency
    return {
        "period_sec": 1.0 / frequency,
        "frequency_hz": frequency,
        "clarity": concentration,
        "spectral_concentration": concentration,
        "estimated_cycles": cycles,
        "reliable": bool(
            concentration >= settings.spectral_concentration_min
            and cycles >= settings.min_cycles
        ),
        "source": "automatic_lagged_ssm",
        "frequency_hz_grid": grid,
        "welch_power": power,
    }, profile


def mask_pca_scores(masks, n_components=3):
    """Return leading PCA score time series without materializing pixel loadings."""
    geometry = mask_geometry(masks)
    gram = np.asarray(geometry["gram"], dtype=np.float64)
    row_mean = gram.mean(axis=1, keepdims=True)
    centered = gram - row_mean - row_mean.T + float(gram.mean())
    centered = 0.5 * (centered + centered.T)

    values, vectors = np.linalg.eigh(centered)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    positive = values > max(float(values[0]) * 1e-12, 1e-12)
    take = min(int(n_components), int(positive.sum()))
    if take == 0:
        return np.empty((len(masks), 0), dtype=np.float64), np.empty(0)
    scores = vectors[:, order[:take]] * np.sqrt(values[:take])[None, :]
    explained = values[:take] / max(float(values.sum()), 1e-12)
    return scores, explained


def period_from_welch(scores, timestamps, settings):
    """Estimate one dominant period from the summed PC-score Welch spectrum."""
    try:
        from scipy.signal import welch
    except ImportError as error:
        raise ImportError("SciPy is required for automatic Welch period estimation") from error

    scores = np.asarray(scores, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    typical_dt = float(np.median(np.diff(timestamps)))
    fps = 1.0 / typical_dt
    duration = float(timestamps[-1] - timestamps[0] + typical_dt)

    if np.max(np.abs(np.diff(timestamps) - typical_dt)) > 0.05 * typical_dt:
        uniform_times = timestamps[0] + np.arange(len(timestamps)) * typical_dt
        scores = np.column_stack(
            [np.interp(uniform_times, timestamps, scores[:, index])
             for index in range(scores.shape[1])]
        )

    n = len(scores)
    nperseg = n if n < 32 else max(16, n // 2)
    noverlap = 0 if nperseg == n else nperseg // 2
    nfft = max(256, 1 << int(np.ceil(np.log2(nperseg))))
    component_power = []
    frequencies = None
    for index in range(scores.shape[1]):
        frequencies, power = welch(
            scores[:, index],
            fs=fps,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend="linear",
            scaling="density",
        )
        component_power.append(power)
    component_power = np.asarray(component_power, dtype=np.float64)
    power = component_power.sum(axis=0)

    minimum_frequency = 1.0 / settings.max_period_sec
    maximum_frequency = min(
        1.0 / settings.min_period_sec,
        settings.welch_nyquist_fraction * fps,
    )
    eligible = (
        (frequencies >= minimum_frequency)
        & (frequencies <= maximum_frequency)
        & np.isfinite(power)
    )
    if not np.any(eligible) or float(power[eligible].sum()) <= 1e-12:
        return {
            "period_sec": None,
            "frequency_hz": None,
            "clarity": None,
            "spectral_concentration": None,
            "estimated_cycles": None,
            "reliable": False,
            "frequency_hz_grid": frequencies,
            "welch_power": power,
            "welch_component_power": component_power,
            "nperseg": int(nperseg),
        }

    eligible_indices = np.flatnonzero(eligible)
    peak = int(eligible_indices[np.argmax(power[eligible])])
    frequency = float(frequencies[peak])
    period = 1.0 / frequency
    cycles = duration * frequency
    lo, hi = max(0, peak - 1), min(len(power), peak + 2)
    concentration = float(power[lo:hi].sum() / power[eligible].sum())
    return {
        "period_sec": period,
        "frequency_hz": frequency,
        "clarity": concentration,
        "spectral_concentration": concentration,
        "estimated_cycles": cycles,
        "reliable": bool(
            concentration >= settings.spectral_concentration_min
            and cycles >= settings.min_cycles
        ),
        "frequency_hz_grid": frequencies,
        "welch_power": power,
        "welch_component_power": component_power,
        "nperseg": int(nperseg),
    }


def estimate_period(masks, timestamps, settings=None):
    """Estimate period with the configured primary frequency estimator."""
    settings = settings or TopologySettings()
    masks, timestamps = validate_sequence(masks, timestamps)
    require_nonempty_masks(masks)
    if settings.frequency_method == "lagged_ssm":
        return period_from_lagged_ssm(masks, timestamps, settings)
    if settings.frequency_method != "welch_pc":
        raise ValueError("frequency_method must be 'lagged_ssm' or 'welch_pc'")
    scores, explained = mask_pca_scores(masks, settings.welch_components)
    if scores.shape[1] == 0:
        return {
            "period_sec": None,
            "frequency_hz": None,
            "clarity": None,
            "spectral_concentration": None,
            "estimated_cycles": None,
            "reliable": False,
            "source": "automatic_welch_pc",
            "n_components_used": 0,
        }, scores
    estimate = period_from_welch(scores, timestamps, settings)
    estimate["source"] = "automatic_welch_pc"
    estimate["n_components_used"] = int(scores.shape[1])
    estimate["pc_variance_ratio"] = explained.tolist()
    return estimate, scores
