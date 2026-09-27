"""Deterministic image and mask perturbations for PRISM stress tests."""


import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, shift


def degrade_sequence(
    frames ,
    masks ,
    *,
    seed  = 0,
    fps_keep_fraction  = 1.0,
    spatial_scale  = 1.0,
    gaussian_blur_sigma  = 0.0,
    pixel_noise_sigma  = 0.0,
    jpeg_quality  = 100,
    mask_boundary_jitter_px  = 0.0,
    mask_flip_probability  = 0.0,
    mask_missing_fraction  = 0.0,
    mask_missing_block_fraction  = 0.0,
    mask_occlusion_fraction  = 0.0,
    mask_morphology_px  = 0,
)    :
    """Apply reproducible degradation and return frames, masks, retained indices."""
    rng = np.random.default_rng(seed)
    frames = np.asarray(frames).copy()
    masks = np.asarray(masks, dtype=bool).copy()

    if 0 < spatial_scale < 1:
        h, w = frames.shape[1:3]
        small = (max(8, round(w * spatial_scale)), max(8, round(h * spatial_scale)))
        for i in range(len(frames)):
            low = cv2.resize(frames[i], small, interpolation=cv2.INTER_AREA)
            frames[i] = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)

    if gaussian_blur_sigma > 0:
        k = max(3, int(np.ceil(gaussian_blur_sigma * 6)) | 1)
        frames = np.asarray([
            cv2.GaussianBlur(frame, (k, k), gaussian_blur_sigma) for frame in frames
        ])

    if pixel_noise_sigma > 0:
        noise = rng.normal(0, pixel_noise_sigma, frames.shape)
        frames = np.clip(frames.astype(float) + noise, 0, 255).astype(np.uint8)

    if jpeg_quality < 100:
        encoded_frames = []
        for frame in frames:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise RuntimeError("JPEG degradation failed")
            encoded_frames.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
        frames = np.asarray(encoded_frames)

    for i, mask in enumerate(masks):
        if mask_boundary_jitter_px > 0:
            dy, dx = rng.normal(0, mask_boundary_jitter_px, 2)
            mask = shift(mask.astype(float), (dy, dx), order=0, mode="constant") > 0.5
        if mask_morphology_px > 0:
            mask = binary_dilation(mask, iterations=mask_morphology_px)
        elif mask_morphology_px < 0:
            mask = binary_erosion(mask, iterations=abs(mask_morphology_px))
        if mask_flip_probability > 0:
            support = binary_dilation(mask, iterations=2)
            flips = (rng.random(mask.shape) < mask_flip_probability) & support
            mask = np.logical_xor(mask, flips)
        if mask_occlusion_fraction > 0:
            width = max(1, round(mask.shape[1] * mask_occlusion_fraction))
            center = round(mask.shape[1] * (0.15 + 0.70 * i / max(len(masks) - 1, 1)))
            mask[:, max(0, center - width // 2): min(mask.shape[1], center + width // 2)] = False
        masks[i] = mask

    valid = np.ones(len(masks), dtype=bool)
    count = min(len(masks) - 8, round(mask_missing_fraction * len(masks)))
    if count > 0:
        valid[rng.choice(len(masks), count, replace=False)] = False
    block = min(len(masks) - 8, round(mask_missing_block_fraction * len(masks)))
    if block > 0:
        start = int(rng.integers(1, max(2, len(masks) - block)))
        valid[start:start + block] = False
    if not valid.all():
        good = np.flatnonzero(valid)
        flat = masks.reshape(len(masks), -1).astype(float)
        for j in np.flatnonzero(np.ptp(flat, axis=0) > 0):
            flat[:, j] = np.interp(np.arange(len(masks)), good, flat[good, j])
        masks = flat.reshape(masks.shape) >= 0.5

    keep_step = max(1, round(1.0 / max(min(fps_keep_fraction, 1.0), 1e-6)))
    retained = np.arange(0, len(frames), keep_step)
    return frames[retained], masks[retained], retained

