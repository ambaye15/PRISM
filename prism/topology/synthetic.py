"""Synthetic SAM-like masks for demonstrations and regression tests."""

import numpy as np


def periodic_translation(
    *,
    n_frames = 120,
    size = 64,
    fps = 10.0,
    period_sec = 2.0,
    amplitude = 9.0,
    jitter = 0.0,
    seed = 7,
):
    """Disk following a closed periodic trajectory, with optional tracking jitter."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_frames, dtype=float) / fps
    yy, xx = np.ogrid[:size, :size]
    masks = []
    for time in times:
        phase = 2.0 * np.pi * time / period_sec
        cx = size / 2 + amplitude * np.sin(phase) + rng.normal(0, jitter)
        cy = size / 2 + 0.35 * amplitude * np.cos(phase) + rng.normal(0, jitter)
        radius = 0.12 * size * (1.0 + 0.04 * np.sin(2.0 * phase))
        masks.append((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2)
    return np.asarray(masks, dtype=bool), times


def amplitude_drift(
    *,
    n_frames = 120,
    size = 64,
    fps = 10.0,
    period_sec = 2.0,
):
    """Periodic motion whose excursion grows gradually through the recording."""
    times = np.arange(n_frames, dtype=float) / fps
    yy, xx = np.ogrid[:size, :size]
    masks = []
    for index, time in enumerate(times):
        phase = 2.0 * np.pi * time / period_sec
        amplitude = 3.0 + 10.0 * index / max(n_frames - 1, 1)
        cx = size / 2 + amplitude * np.sin(phase)
        cy = size / 2 + 0.25 * amplitude * np.cos(phase)
        masks.append((xx - cx) ** 2 + (yy - cy) ** 2 <= (0.12 * size) ** 2)
    return np.asarray(masks, dtype=bool), times


def aperiodic_random_walk(
    *,
    n_frames = 120,
    size = 64,
    fps = 10.0,
    seed = 11,
):
    """Smooth but nonperiodic motion used as a negative control."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_frames, dtype=float) / fps
    steps = rng.normal(0.0, 0.9, size=(n_frames, 2))
    path = np.cumsum(steps, axis=0)
    path -= path.mean(axis=0)
    maximum = np.maximum(np.max(np.abs(path), axis=0), 1.0)
    path = 11.0 * path / maximum
    yy, xx = np.ogrid[:size, :size]
    masks = []
    for dx, dy in path:
        cx, cy = size / 2 + dx, size / 2 + dy
        masks.append((xx - cx) ** 2 + (yy - cy) ** 2 <= (0.12 * size) ** 2)
    return np.asarray(masks, dtype=bool), times


def periodic_articulation(
    *,
    n_frames = 120,
    size = 64,
    fps = 10.0,
    period_sec = 1.5,
):
    """Stationary torso with two periodically moving arm-like components."""
    times = np.arange(n_frames, dtype=float) / fps
    yy, xx = np.ogrid[:size, :size]
    masks = []
    cx, cy = size / 2, 0.53 * size
    torso = ((xx - cx) / 7.0) ** 2 + ((yy - cy) / 13.0) ** 2 <= 1.0
    head = (xx - cx) ** 2 + (yy - (cy - 17)) ** 2 <= 5.0 ** 2
    for time in times:
        phase = 2.0 * np.pi * time / period_sec
        lift = 8.0 * np.sin(phase)
        left = ((xx - (cx - 11)) / 7.5) ** 2 + ((yy - (cy - lift)) / 3.0) ** 2 <= 1.0
        right = ((xx - (cx + 11)) / 7.5) ** 2 + ((yy - (cy + lift)) / 3.0) ** 2 <= 1.0
        masks.append(torso | head | left | right)
    return np.asarray(masks, dtype=bool), times

