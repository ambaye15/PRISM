import numpy as np


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / length


def look_at_camera(azimuth_deg, elevation_deg, distance=4.0, focal_px=900.0,
                   image_size=(1280, 720)):
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    center = np.array([
        distance * np.cos(elevation) * np.sin(azimuth),
        distance * np.sin(elevation),
        distance * np.cos(elevation) * np.cos(azimuth),
    ])
    forward = normalize(-center)
    provisional_up = np.array([0.0, 1.0, 0.0])
    right = normalize(np.cross(forward, provisional_up))
    up = normalize(np.cross(right, forward))
    rotation = np.stack([right, up, forward], axis=0)
    translation = -rotation @ center
    width, height = image_size
    intrinsic = np.array([
        [focal_px, 0.0, width / 2.0],
        [0.0, focal_px, height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    projection = intrinsic @ np.column_stack([rotation, translation])
    return {
        "projection": projection,
        "rotation": rotation,
        "translation": translation,
        "center": center,
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
    }


def project(points, projection):
    points = np.asarray(points, dtype=float)
    homogeneous = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1))], axis=-1
    )
    image = homogeneous @ np.asarray(projection, dtype=float).T
    valid = image[..., 2] > 1e-8
    xy = np.full(image.shape[:-1] + (2,), np.nan)
    xy[valid] = image[valid, :2] / image[valid, 2, None]
    return xy, valid


def triangulate_point(observations, projections):
    rows = []
    for point, projection in zip(observations, projections):
        if not np.all(np.isfinite(point)):
            continue
        x, y = point
        projection = np.asarray(projection, dtype=float)
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
    if len(rows) < 4:
        return np.full(3, np.nan)
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    value = vh[-1]
    if abs(value[3]) <= 1e-12:
        return np.full(3, np.nan)
    return value[:3] / value[3]


def triangulate_sequence(observations, projections):
    observations = np.asarray(observations, dtype=float)
    frames, cameras, joints, _ = observations.shape
    reconstructed = np.full((frames, joints, 3), np.nan)
    for frame in range(frames):
        for joint in range(joints):
            reconstructed[frame, joint] = triangulate_point(
                observations[frame, :, joint], projections
            )
    return reconstructed


def add_observation_noise(points, sigma_px, missing_probability, rng):
    output = np.asarray(points, dtype=float).copy()
    finite = np.isfinite(output).all(axis=-1)
    output[finite] += rng.normal(0.0, sigma_px, size=(finite.sum(), 2))
    missing = rng.random(finite.shape) < missing_probability
    output[missing] = np.nan
    return output

