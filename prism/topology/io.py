"""Input/output helpers for standalone SAM-mask topology."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

MASK_KEYS = ("masks", "sam_masks", "mask", "masks_fixed")
TIME_KEYS = ("timestamps_sec", "timestamps", "times_sec", "times")
VALID_KEYS = ("mask_valid", "valid_mask", "frame_valid_mask")


def _binary_masks(array, threshold):
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"expected TxHxW masks, found shape {array.shape}")
    if array.dtype == bool:
        return array.copy()
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmin(array)) < 0.0:
        return array > 0.0
    return array > threshold


def load_masks(
    path,
    *,
    fps = None,
    threshold = 0.5,
):
    """Load masks from NPZ, NPY, or a directory of ordered mask images.

    NPZ files may include ``timestamps_sec`` and ``mask_valid``.  If timestamps
    are absent, ``fps`` is required.  Image directories accept PNG, TIFF, BMP,
    or JPEG files sorted lexicographically.
    """
    source = Path(path).expanduser().resolve()
    timestamps = None
    valid = None
    selected_key = None
    if source.is_dir():
        extensions = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in extensions)
        if not files:
            raise FileNotFoundError(f"no mask images found in {source}")
        masks = np.stack([np.asarray(Image.open(p).convert("L")) for p in files])
        metadata = {"format": "image_directory", "n_files": len(files)}
    elif source.suffix.lower() == ".npy":
        masks = np.load(source, allow_pickle=False)
        metadata = {"format": "npy"}
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as saved:
            selected_key = next((key for key in MASK_KEYS if key in saved.files), None)
            if selected_key is None:
                raise KeyError(
                    f"NPZ needs one mask key from {MASK_KEYS}; found {saved.files}"
                )
            masks = saved[selected_key]
            time_key = next((key for key in TIME_KEYS if key in saved.files), None)
            valid_key = next((key for key in VALID_KEYS if key in saved.files), None)
            if time_key is not None:
                timestamps = saved[time_key].astype(np.float64)
            if valid_key is not None:
                valid = saved[valid_key].astype(bool)
        metadata = {"format": "npz", "mask_key": selected_key}
    else:
        raise ValueError("mask input must be .npz, .npy, or an image directory")

    masks = _binary_masks(masks, threshold)
    if timestamps is None:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("provide --fps when the input has no timestamps")
        timestamps = np.arange(len(masks), dtype=np.float64) / float(fps)
    if len(timestamps) != len(masks):
        raise ValueError("timestamp count differs from mask count")
    observed = masks.reshape(len(masks), -1).any(axis=1)
    valid = observed if valid is None else np.asarray(valid, dtype=bool) & observed
    if len(valid) != len(masks):
        raise ValueError("valid-mask count differs from mask count")
    metadata.update(
        {
            "source": str(source),
            "n_frames": len(masks),
            "height": int(masks.shape[1]),
            "width": int(masks.shape[2]),
            "threshold": float(threshold),
        }
    )
    return masks, timestamps, valid, metadata


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save_analysis(
    result,
    output_dir,
    *,
    input_metadata = None,
    title = "SAM mask sliding-window topology",
    create_figure = True,
):
    """Save summary JSON, numerical NPZ, and one diagnostic figure."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(result["summary"])
    if input_metadata is not None:
        summary["input_metadata"] = input_metadata
    summary_path = output / "topology_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    arrays_path = output / "topology_arrays.npz"
    arrays = {
        key: value for key, value in result.items()
        if key != "summary" and value is not None
    }
    np.savez_compressed(arrays_path, **arrays)

    files = {
        "summary": str(summary_path),
        "arrays": str(arrays_path),
        "figure": None,
    }
    if not create_figure:
        return files

    from ..plotting.topology import plot_topology_diagnostic

    figure_path = plot_topology_diagnostic(result, summary, output, title)
    files["figure"] = str(figure_path)
    return files
