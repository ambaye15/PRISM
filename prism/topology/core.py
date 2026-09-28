import numpy as np

from .._records import asdict
from .embedding import knn_density_selection, sliding_window_distances
from .masks import TopologySettings, longest_valid_run, require_nonempty_masks, validate_sequence
from .period import estimate_period


def pca_from_distances(distance, n_components = 3):
    """Classical MDS, equivalent to PCA scores for Euclidean distances."""
    distance = np.asarray(distance, dtype=np.float64)
    squared = distance * distance
    gram = -0.5 * (
        squared
        - squared.mean(axis=0, keepdims=True)
        - squared.mean(axis=1, keepdims=True)
        + float(squared.mean())
    )
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    take = min(n_components, len(values))
    coordinates = vectors[:, :take] * np.sqrt(values[:take])[None, :]
    total = float(values.sum())
    ratios = values[:take] / total if total > 1e-12 else np.zeros(take)
    return coordinates, ratios


def orbit_radius_from_distances(distance):
    """RMS distance to the cloud centroid from a Euclidean distance matrix.

    The identity ``r_rms^2 = mean_{i,j}(D_ij^2) / 2`` avoids choosing an
    arbitrary coordinate system for the sliding-window point cloud.
    """
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("distance must contain finite nonnegative values")
    return float(np.sqrt(np.mean(distance * distance) / 2.0))


def periodicity_score_from_diagram(diagram):
    """Dominant-loop periodicity with a birth-weighted second-loop penalty."""
    diagram = np.asarray(diagram, dtype=np.float64)
    if diagram.size == 0:
        return 0.0
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    if not len(finite):
        return 0.0
    persistence = finite[:, 1] - finite[:, 0]
    order = np.argsort(persistence)[::-1]
    first = finite[order[0]]
    first_term = np.exp(-first[0] / 2.0) * (first[1] - first[0]) / np.sqrt(3.0)
    second_term = 0.0
    if len(order) > 1:
        second = finite[order[1]]
        second_term = (
            np.exp(-second[0] / 2.0)
            * (second[1] - second[0])
            / np.sqrt(3.0)
        )
    return float(first_term - second_term)

def persistence_summary(distance, compute = True, *, normalized_cloud = False):
    """Compute finite H1 persistence with Ripser."""
    empty = np.empty((0, 2), dtype=np.float64)
    if not compute:
        return {"h1_computed": False}, empty
    try:
        from ripser import ripser
    except ImportError as error:
        raise ImportError(
            "ripser is required for H1 computation; install it with "
            "`python -m pip install -e \".[analysis]\"`"
        ) from error
    diagram = np.asarray(ripser(distance, distance_matrix=True, maxdim=1, coeff = 47)["dgms"][1], dtype=np.float64)
    diagram = diagram[np.isfinite(diagram).all(axis=1)] if diagram.size else empty
    persistence = diagram[:, 1] - diagram[:, 0] if diagram.size else np.asarray([])
    diameter = float(np.max(distance))
    maximum = float(persistence.max()) if persistence.size else 0.0
    return {
        "h1_computed": True,
        "h1_max_persistence": maximum,
        "h1_total_persistence": float(persistence.sum()) if persistence.size else 0.0,
        "h1_relative_persistence": maximum / diameter if diameter > 1e-12 else 0.0,
        "periodicity_score": (
            periodicity_score_from_diagram(diagram)
            if normalized_cloud
            else None
        ),
        "point_cloud_diameter": diameter,
        "point_cloud_orbit_radius": orbit_radius_from_distances(distance),
    }, diagram

def cycle_return_distance(distance, starts, period):
    """Median distance between windows one period apart."""
    values = []
    tolerance = max(float(np.median(np.diff(starts))) * 1.5, 1e-8)
    for index, time in enumerate(starts):
        target = time + period
        match = int(np.argmin(np.abs(starts - target)))
        if abs(starts[match] - target) <= tolerance:
            values.append(float(distance[index, match]))
    return float(np.median(values)) if values else None

def resolve_period(masks, timestamps, settings):
    """Estimated or user-supplied period, with PCA scores behind the estimate."""
    if settings.period_sec is None:
        period, period_signals = estimate_period(masks, timestamps, settings)
        if period["period_sec"] is None:
            raise ValueError("automatic period estimation failed; supply --period-sec")
        return period, period_signals
    value = float(settings.period_sec)
    cycles = (timestamps[-1] - timestamps[0]) / value
    period = {
        "period_sec": value,
        "frequency_hz": 1.0 / value,
        "clarity": None,
        "estimated_cycles": cycles,
        "reliable": bool(cycles >= settings.min_cycles),
        "source": "user_supplied",
    }
    return period, None

def analyze_masks(masks, timestamps, *, valid_mask = None, settings = None):
    """Complete analysis returning JSON-friendly summaries and NumPy arrays."""
    settings = settings or TopologySettings()
    masks, timestamps = validate_sequence(masks, timestamps)
    observed = masks.reshape(len(masks), -1).any(axis=1)
    if valid_mask is None:
        valid_mask = observed
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if len(valid_mask) != len(masks):
            raise ValueError("valid_mask and masks must contain the same number of frames")
        valid_mask = valid_mask & observed
    run = longest_valid_run(valid_mask)
    masks, timestamps = masks[run], timestamps[run]
    require_nonempty_masks(masks)

    period, period_signals = resolve_period(masks, timestamps, settings)
    cloud = sliding_window_distances(
        masks,
        timestamps,
        period_sec=float(period["period_sec"]),
        d=settings.d,
        n_points=settings.n_points,
    )

    density = knn_density_selection(
        cloud["scale_distance"],
        k=settings.density_k,
        retain_fraction=settings.density_retain_fraction,
    )

    # Density selection is defined on the complete, unnormalized scale cloud.
    scale_keep = np.flatnonzero(density["keep_mask"])
    starts = cloud["start_times_sec"][scale_keep]
    scale = cloud["scale_distance"][np.ix_(scale_keep, scale_keep)]

    # Zero-variation windows are absent from the sphere-normalized cloud.
    # Convert the original density mask into indices for this reduced cloud.
    sphere_valid = cloud["sphere_valid_mask"]
    normalized_density_keep = density["keep_mask"][sphere_valid]
    normalized_keep = np.flatnonzero(normalized_density_keep)

    if len(normalized_keep) < 8:
        raise ValueError(
            "fewer than 8 sphere-normalized windows remain after "
            "density selection"
        )

    normalized_starts = cloud["normalized_start_times_sec"][
        normalized_keep
    ]
    normalized = cloud["normalized_distance"][
        np.ix_(normalized_keep, normalized_keep)
    ]

    scale_pca, scale_ratio = pca_from_distances(scale)
    normalized_pca, normalized_ratio = pca_from_distances(
        normalized
    )

    scale_h1, scale_diagram = persistence_summary(
        scale,
        settings.compute_h1,
        normalized_cloud=False,
    )
    normalized_h1, normalized_diagram = persistence_summary(
        normalized,
        settings.compute_h1,
        normalized_cloud=True,
    )

    summary = {
        "settings": asdict(settings),
        "input": {
            "n_frames_original": len(valid_mask),
            "n_frames_analyzed": len(run),
            "analyzed_start_index": int(run[0]),
            "analyzed_end_index_exclusive": int(run[-1] + 1),
            "height": int(masks.shape[1]),
            "width": int(masks.shape[2]),
        },
        "period": {
            k: v
            for k, v in period.items()
            if k not in {
                "frequency_hz_grid",
                "welch_power",
                "welch_component_power",
            }
        },
        "embedding": {
            "d": settings.d,
            "window_samples": settings.d + 1,
            "tau_sec": cloud["tau_sec"],
            "window_span_sec": cloud["window_span_sec"],
            "n_points_before_filter": settings.n_points,

            # Keep the original key for backward compatibility.
            "n_points_retained": len(scale_keep),

            "n_scale_points_retained": len(scale_keep),
            "n_normalized_points_before_density": int(
                cloud["n_sphere_points"]
            ),
            "n_normalized_points_retained": len(normalized_keep),
            "sphere_valid_fraction": float(
                cloud["sphere_valid_fraction"]
            ),
            "centering": (
                "pixelwise temporal mean across d+1 masks"
            ),
            "normalization": (
                "L2 sphere normalization after temporal image "
                "centering; zero-variation windows excluded"
            ),
        },
        "density": {
            k: v
            for k, v in density.items()
            if k not in {
                "keep_mask",
                "density",
                "knn_radius",
            }
        },
        "scale": {
            **scale_h1,
            "pc_variance_ratio": scale_ratio.tolist(),
        },
        "normalized": {
            **normalized_h1,
            "pc_variance_ratio": normalized_ratio.tolist(),
            "cycle_return_distance_median": (
                cycle_return_distance(
                    normalized,
                    normalized_starts,
                    float(period["period_sec"]),
                )
            ),
            "pre_sphere_norm_min": float(
                cloud["pre_sphere_norm"][
                    normalized_keep
                ].min()
            ),
            "pre_sphere_norm_median": float(
                np.median(
                    cloud["pre_sphere_norm"][
                        normalized_keep
                    ]
                )
            ),
            "pre_sphere_norm_max": float(
                cloud["pre_sphere_norm"][
                    normalized_keep
                ].max()
            ),
        },
    }

    return {
        "summary": summary,

        # Backward-compatible scale-cloud timestamps.
        "start_times_sec": starts,
        "start_times_sec_all": cloud["start_times_sec"],

        # Explicit normalized-cloud timestamps and validity map.
        "normalized_start_times_sec": normalized_starts,
        "normalized_start_times_sec_all": cloud[
            "normalized_start_times_sec"
        ],
        "sphere_valid_mask": sphere_valid,

        "density_keep_mask": density["keep_mask"],
        "density": density["density"],
        "scale_distance": scale,
        "normalized_distance": normalized,
        "scale_pca": scale_pca,
        "normalized_pca": normalized_pca,
        "scale_diagram": scale_diagram,
        "normalized_diagram": normalized_diagram,
        "frequency_hz_grid": period.get(
            "frequency_hz_grid"
        ),
        "frequency_power": period.get("welch_power"),
        "welch_component_power": period.get(
            "welch_component_power"
        ),
        "period_pca_scores": period_signals,
    }