import csv
import json
import re
from pathlib import Path

import numpy as np

LABELS = {
    0: "none",
    1: "rock",
    2: "flap",
    3: "flap_rock",
}


def safe_id(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not value:
        raise ValueError("identifier cannot be empty")
    return value


def read_manifest(path, project_root = None):
    path = Path(path).expanduser().resolve()
    project_root = Path(project_root or path.parent.parent).expanduser().resolve()
    payload = json.loads(path.read_text(encoding = "utf-8"))
    videos = payload.get("videos") if isinstance(payload, dict) else payload
    if not isinstance(videos, list) or not videos:
        raise ValueError("manifest must contain a nonempty videos list")

    result = []
    seen = set()
    for index, raw in enumerate(videos):
        if not isinstance(raw, dict):
            raise ValueError("video %d is not an object" % index)
        entry = dict(raw)
        raw_path = entry.get("path", entry.get("video_path"))
        if raw_path is None:
            raise ValueError("video %d has no path" % index)
        video_path = Path(raw_path).expanduser()
        if not video_path.is_absolute():
            choices = [project_root / video_path, path.parent / video_path]
            video_path = next(
                (choice for choice in choices if choice.is_file()),
                choices[0],
            )
        video_path = video_path.resolve()
        video_id = safe_id(
            entry.get("video_id", entry.get("name", video_path.stem))
        )
        if video_id in seen:
            raise ValueError("duplicate video_id: %s" % video_id)
        seen.add(video_id)

        for key in ("participant_id", "study_id", "session_id", "fps"):
            if entry.get(key) is None:
                raise ValueError("video %s is missing %s" % (video_id, key))
        annotations = load_annotations(entry, path.parent)
        if len(annotations) == 0:
            raise ValueError("video %s has no annotations" % video_id)
        unknown = sorted(set(np.unique(annotations)).difference(LABELS))
        if unknown:
            raise ValueError(
                "video %s contains unsupported labels: %s" %
                (video_id, unknown)
            )
        fps = float(entry["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video %s has invalid fps" % video_id)

        entry.update({
            "video_id": video_id,
            "path": video_path,
            "fps": fps,
            "annotations": annotations,
            "annotation_start_frame": int(entry.get("annotation_start_frame", 0)),
        })
        result.append(entry)
    return result


def load_annotations(entry, manifest_directory):
    if "annotations" in entry:
        values = entry["annotations"]
    elif "annotations_path" in entry:
        path = Path(entry["annotations_path"]).expanduser()
        if not path.is_absolute():
            path = manifest_directory / path
        suffix = path.suffix.lower()
        if suffix == ".npy":
            values = np.load(path, allow_pickle = False)
        elif suffix == ".json":
            values = json.loads(path.read_text(encoding = "utf-8"))
        else:
            raise ValueError("annotations_path must end in .npy or .json")
    else:
        raise ValueError("each video needs annotations or annotations_path")
    values = np.asarray(values)
    if values.ndim != 1:
        values = values.reshape(-1)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("annotations must be numeric labels")
    if np.any(~np.isfinite(values.astype(float))):
        raise ValueError("annotations contain nonfinite values")
    rounded = np.round(values).astype(int)
    if np.any(values.astype(float) != rounded):
        raise ValueError("annotations must contain integer labels")
    return rounded


def extract_bouts(annotations, start_frame = 0):
    annotations = np.asarray(annotations, dtype = int).reshape(-1)
    bouts = []
    start = None
    label = 0
    for index in range(len(annotations) + 1):
        current = int(annotations[index]) if index < len(annotations) else 0
        if current != label:
            if label != 0:
                bouts.append({
                    "label": label,
                    "behavior": LABELS[label],
                    "start_frame": int(start_frame + start),
                    "end_frame": int(start_frame + index),
                })
            start = index if current != 0 else None
            label = current
    return bouts


def inspect_video(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError("could not open video: %s" % path)
    result = {
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    }
    capture.release()
    if result["frame_count"] < 1 or result["fps"] <= 0:
        raise ValueError("video reports invalid frame count or fps: %s" % path)
    return result


def write_clip(video_path, output_path, start_frame, end_frame, fps):
    import cv2

    output_path.parent.mkdir(parents = True, exist_ok = True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError("could not open video: %s" % video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("could not decode frame %d" % start_frame)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("could not create clip: %s" % output_path)
    writer.write(frame)
    count = 1
    while start_frame + count < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        count += 1
    writer.release()
    capture.release()
    expected = end_frame - start_frame
    if count != expected:
        output_path.unlink(missing_ok = True)
        raise RuntimeError(
            "decoded %d frames but expected %d from %s" %
            (count, expected, video_path)
        )


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", newline = "", encoding = "utf-8") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames = keys)
        writer.writeheader()
        writer.writerows(rows)


def extract_dataset(
    manifest,
    output,
    project_root = None,
    context_sec = 1.0,
    limit = None,
    force = False,
    allow_fps_mismatch = False,
):
    output = Path(output).expanduser().resolve()
    videos = read_manifest(manifest, project_root = project_root)
    inventory = []
    episodes = []
    session_rows = []

    for video in videos:
        info = inspect_video(video["path"])
        annotation_stop = video["annotation_start_frame"] + len(video["annotations"])
        if annotation_stop > info["frame_count"]:
            raise ValueError(
                "video %s has %d annotated frames ending at %d, but the video has %d" %
                (
                    video["video_id"],
                    len(video["annotations"]),
                    annotation_stop,
                    info["frame_count"],
                )
            )
        tolerance = max(0.05, 0.01 * video["fps"])
        if abs(info["fps"] - video["fps"]) > tolerance and not allow_fps_mismatch:
            raise ValueError(
                "manifest fps %.6g differs from video fps %.6g for %s" %
                (video["fps"], info["fps"], video["video_id"])
            )
        bouts = extract_bouts(
            video["annotations"],
            start_frame = video["annotation_start_frame"],
        )
        inventory.append({
            "video_id": video["video_id"],
            "path": str(video["path"]),
            "participant_id": video["participant_id"],
            "study_id": video["study_id"],
            "session_id": video["session_id"],
            "manifest_fps": video["fps"],
            "video_fps": info["fps"],
            "frame_count": info["frame_count"],
            "annotation_count": len(video["annotations"]),
            "n_bouts": len(bouts),
        })
        session = {
            "participant_id": video["participant_id"],
            "study_id": video["study_id"],
            "session_id": video["session_id"],
            "video_id": video["video_id"],
            "recorded_duration_sec": len(video["annotations"]) / video["fps"],
            "n_bouts": len(bouts),
        }
        for behavior in ("rock", "flap", "flap_rock"):
            selected = [bout for bout in bouts if bout["behavior"] == behavior]
            frames = sum(bout["end_frame"] - bout["start_frame"] for bout in selected)
            session["%s_n_bouts" % behavior] = len(selected)
            session["%s_duration_sec" % behavior] = frames / video["fps"]
        session_rows.append(session)

        counters = {name: 0 for name in ("rock", "flap", "flap_rock")}
        context_frames = int(round(context_sec * video["fps"]))
        for bout in bouts:
            counters[bout["behavior"]] += 1
            episode_id = (
                "%s__%s_%04d__f%07d-%07d" %
                (
                    video["video_id"],
                    bout["behavior"],
                    counters[bout["behavior"]],
                    bout["start_frame"],
                    bout["end_frame"] - 1,
                )
            )
            analysis_start = max(0, bout["start_frame"] - context_frames)
            analysis_end = min(info["frame_count"], bout["end_frame"] + context_frames)
            row = {
                "episode_id": episode_id,
                "video_id": video["video_id"],
                "participant_id": video["participant_id"],
                "study_id": video["study_id"],
                "session_id": video["session_id"],
                "setting": video.get("setting"),
                "behavior": bout["behavior"],
                "label": bout["label"],
                "start_frame": bout["start_frame"],
                "end_frame": bout["end_frame"],
                "core_start_frame": bout["start_frame"],
                "core_end_frame": bout["end_frame"],
                "analysis_start_frame": analysis_start,
                "analysis_end_frame_exclusive": analysis_end,
                "start_sec": bout["start_frame"] / video["fps"],
                "end_sec": bout["end_frame"] / video["fps"],
                "duration_sec": (bout["end_frame"] - bout["start_frame"]) / video["fps"],
                "fps": video["fps"],
                "source_video_path": str(video["path"]),
            }
            for key in ("recording_date", "camera_id", "notes"):
                if video.get(key) is not None:
                    row[key] = video[key]
            episodes.append(row)

    selected = episodes if limit is None else episodes[:int(limit)]
    for index, row in enumerate(selected, start = 1):
        print("[%d/%d] %s" % (index, len(selected), row["episode_id"]))
        episode_dir = output / "episode_runs" / row["episode_id"]
        assets = episode_dir / "assets"
        clip = assets / (row["episode_id"] + "_raw.mp4")
        episode_dir.mkdir(parents = True, exist_ok = True)
        (episode_dir / "episode.json").write_text(
            json.dumps(row, indent = 2, sort_keys = True) + "\n",
            encoding = "utf-8",
        )
        if force or not clip.is_file():
            write_clip(
                row["source_video_path"],
                clip,
                row["analysis_start_frame"],
                row["analysis_end_frame_exclusive"],
                row["fps"],
            )

    output.mkdir(parents = True, exist_ok = True)
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "annotation_session_summary.csv", session_rows)
    report = {
        "schema_version": "prism_annotated_dataset.v1",
        "n_videos": len(videos),
        "n_episodes": len(episodes),
        "n_selected": len(selected),
        "n_processed": len(selected),
        "output_dir": str(output),
        "videos": inventory,
    }
    (output / "dataset_inventory.json").write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8",
    )
    return report

