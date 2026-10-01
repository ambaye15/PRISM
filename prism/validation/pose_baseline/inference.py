"""Pose inference on crops derived from the tracked DAM4SAM mask."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def mask_box(mask, padding_fraction=0.15):
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    height, width = mask.shape
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    padding = int(round(max(x1 - x0, y1 - y0) * float(padding_fraction)))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def box_iou(box, target):
    x0 = max(float(box[0]), float(target[0]))
    y0 = max(float(box[1]), float(target[1]))
    x1 = min(float(box[2]), float(target[2]))
    y1 = min(float(box[3]), float(target[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = max(0.0, float(target[2] - target[0])) * max(0.0, float(target[3] - target[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class UltralyticsPose:
    def __init__(self, model="yolo11m-pose.pt", device=None, image_size=640):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Pose inference requires Ultralytics. Install with "
                "`pip install ultralytics` or `pip install -e '.[pose]'`."
            ) from error
        self.model_name = model
        self.model = YOLO(model)
        self.device = device
        self.image_size = int(image_size)

    def predict(self, crop, local_mask):
        result = self.model.predict(
            source=crop,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        xy = result.keypoints.xy.detach().cpu().numpy()
        if result.keypoints.conf is None:
            confidence = np.ones(xy.shape[:2], dtype=np.float64)
        else:
            confidence = result.keypoints.conf.detach().cpu().numpy()

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        target = mask_box(local_mask, padding_fraction=0.0)
        if target is None:
            return None
        scores = [box_iou(box, target) for box in boxes]
        selected = int(np.argmax(scores))
        return xy[selected], confidence[selected], float(scores[selected])


def read_bout_inputs(episode_dir, source=None, videos_root=None):
    episode_dir = Path(episode_dir)
    npz_path = episode_dir / "sam2_masks_fixed_crop.npz"
    summary_path = episode_dir / "subject_dino_summary.json"
    if not npz_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("missing bout masks or subject summary in " + str(episode_dir))
    data = np.load(npz_path)
    masks = np.asarray(data["masks"], dtype=bool)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_clip = Path(summary["source_cached_video"]).expanduser()
    video_start_frame = 0
    if not raw_clip.is_file():
        source_video = None if source is None else source.get("source_video_path")
        if source_video is None or str(source_video).strip() in {"", "nan", "None"}:
            raise FileNotFoundError(
                "raw bout clip not found and source_video_path is unavailable: "
                + str(raw_clip)
            )
        raw_clip = Path(str(source_video)).expanduser()
        if not raw_clip.is_file() and videos_root is not None:
            raw_clip = Path(videos_root).expanduser() / raw_clip.name
        if not raw_clip.is_file():
            raise FileNotFoundError(
                "raw bout clip and full source video were not found: " + str(raw_clip)
            )
        start = source.get("analysis_start_frame", 0)
        video_start_frame = int(start) if np.isfinite(float(start)) else 0
    camera_transforms, camera_translations = read_camera_corrections(
        episode_dir / "optical_flow_frames.csv", len(masks) - 1
    )
    return {
        "masks": masks,
        "mask_valid": np.asarray(data["mask_valid"], dtype=bool),
        "core_mask": np.asarray(data["core_mask"], dtype=bool),
        "timestamps_sec": np.asarray(data["timestamps_sec"], dtype=np.float64),
        "raw_clip": raw_clip,
        "video_start_frame": video_start_frame,
        "summary": summary,
        "camera_transforms": camera_transforms,
        "camera_translations": camera_translations,
    }


def number_or_none(value):
    if value is None or str(value).strip() in {"", "None", "nan"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def read_camera_corrections(path, pair_count):
    """Read the exact corrections used by the frozen optical-flow analysis."""
    transforms = [None] * pair_count
    translations = [None] * pair_count
    if not path.is_file():
        return transforms, translations
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                index = int(row["frame_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if index < 0 or index >= pair_count:
                continue
            encoded = row.get("flow_camera_transform")
            if encoded and encoded not in {"None", "nan"}:
                try:
                    matrix = np.asarray(json.loads(encoded), dtype=np.float64)
                    if matrix.shape == (2, 3) and np.isfinite(matrix).all():
                        transforms[index] = matrix
                        continue
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
            dx = number_or_none(row.get("flow_camera_translation_x_pixels_per_frame"))
            dy = number_or_none(row.get("flow_camera_translation_y_pixels_per_frame"))
            if dx is not None and dy is not None:
                translations[index] = np.asarray([dx, dy], dtype=np.float64)
    return transforms, translations


def infer_bout_pose(inputs, estimator, padding_fraction=0.15):
    masks = inputs["masks"]
    mask_valid = inputs["mask_valid"]
    capture = cv2.VideoCapture(str(inputs["raw_clip"]))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open " + str(inputs["raw_clip"]))
    start_frame = int(inputs.get("video_start_frame", 0))
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    xy_rows = []
    confidence_rows = []
    overlap_rows = []
    joint_count = 17
    for index in range(len(masks)):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError("video ended before frame " + str(index))
        if not mask_valid[index]:
            xy_rows.append(np.full((joint_count, 2), np.nan))
            confidence_rows.append(np.full(joint_count, np.nan))
            overlap_rows.append(np.nan)
            continue
        box = mask_box(masks[index], padding_fraction)
        if box is None:
            xy_rows.append(np.full((joint_count, 2), np.nan))
            confidence_rows.append(np.full(joint_count, np.nan))
            overlap_rows.append(np.nan)
            continue
        x0, y0, x1, y1 = box
        crop = frame[y0:y1, x0:x1]
        local_mask = masks[index, y0:y1, x0:x1]
        prediction = estimator.predict(crop, local_mask)
        if prediction is None:
            xy_rows.append(np.full((joint_count, 2), np.nan))
            confidence_rows.append(np.full(joint_count, np.nan))
            overlap_rows.append(np.nan)
            continue
        xy, confidence, overlap = prediction
        joint_count = len(xy)
        xy = np.asarray(xy, dtype=np.float64)
        xy[:, 0] += x0
        xy[:, 1] += y0
        xy_rows.append(xy)
        confidence_rows.append(np.asarray(confidence, dtype=np.float64))
        overlap_rows.append(overlap)
    capture.release()
    return (
        np.asarray(xy_rows, dtype=np.float64),
        np.asarray(confidence_rows, dtype=np.float64),
        np.asarray(overlap_rows, dtype=np.float64),
    )
