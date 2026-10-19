import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "prism.dam4sam.bout_adapter.v1"


def parse_args(argv    = None)  :
    parser = argparse.ArgumentParser(
        description="Prepare bout-level PRISM inputs from full-video DAM4SAM masks."
    )
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def read_json(path )   :
    return json.loads(path.read_text(encoding="utf-8"))


def find_raw_clip(episode_dir )  :
    assets = episode_dir / "assets"
    matches = sorted(assets.glob("*_raw.mp4")) if assets.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_raw.mp4 in {assets}, found {len(matches)}"
        )
    return matches[0].resolve()


def inspect_clip(path )     :
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is unavailable; activate dam4sam_env") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    result = {
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    }
    capture.release()
    return result


def episode_interval(metadata  )     :
    required = (
        "analysis_start_frame",
        "analysis_end_frame_exclusive",
        "core_start_frame",
        "core_end_frame",
    )
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise KeyError("Episode metadata is missing: " + ", ".join(missing))
    analysis_start = int(metadata["analysis_start_frame"])
    analysis_end = int(metadata["analysis_end_frame_exclusive"])
    core_start = int(metadata["core_start_frame"])
    core_end = int(metadata["core_end_frame"])
    if not analysis_start <= core_start < core_end <= analysis_end:
        raise ValueError(
            "Invalid analysis/core intervals: "
            f"analysis=[{analysis_start}, {analysis_end}), "
            f"core=[{core_start}, {core_end})"
        )
    return analysis_start, analysis_end, core_start, core_end


class VideoTrack:
    def __init__(self, root , video_id ):
        directory = root / video_id
        summary_path = directory / "dam4sam_track_summary.json"
        packed_path = directory / "dam4sam_video_masks_packed.npy"
        valid_path = directory / "dam4sam_video_mask_valid.npy"
        for path in (summary_path, packed_path, valid_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing DAM4SAM output: {path}")

        self.summary = read_json(summary_path)
        if self.summary.get("video_id") != video_id:
            raise ValueError(f"Track summary video ID mismatch for {video_id}")
        self.width = int(self.summary["frame_width"])
        self.height = int(self.summary["frame_height"])
        self.frame_count = int(self.summary["frame_count"])
        self.fps = float(self.summary["fps"])
        self.packed = np.load(packed_path, mmap_mode="r")
        self.valid = np.load(valid_path, mmap_mode="r")
        expected_shape = (
            self.frame_count,
            self.height,
            (self.width + 7) // 8,
        )
        if self.packed.shape != expected_shape:
            raise ValueError(
                f"Packed mask shape {self.packed.shape} != {expected_shape}"
            )
        if self.valid.shape != (self.frame_count,):
            raise ValueError(
                f"Validity shape {self.valid.shape} != {(self.frame_count,)}"
            )

    def slice(
        self, start , end     )    :
        if not 0 <= start < end <= self.frame_count:
            raise IndexError(
                f"Requested source interval [{start}, {end}) outside "
                f"[0, {self.frame_count})"
            )
        packed = np.asarray(self.packed[start:end], dtype=np.uint8)
        masks = np.unpackbits(packed, axis=-1)[..., : self.width].astype(bool)
        valid = np.asarray(self.valid[start:end], dtype=bool)
        actual_nonempty = masks.reshape(len(masks), -1).any(axis=1)
        if not np.array_equal(valid, actual_nonempty):
            mismatches = int(np.count_nonzero(valid != actual_nonempty))
            raise ValueError(
                f"Mask validity disagrees with mask contents on {mismatches} frames"
            )
        return packed, masks, valid


def write_npz_atomic(path , **arrays )  :
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_csv(path , rows  )  :
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not keys:
            return
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare_episode(
    episode_dir ,
    output_root ,
    tracks_root ,
    track_cache  ,
    *,
    force ,
)   :
    metadata_path = episode_dir / "episode.json"
    metadata = read_json(metadata_path)
    episode_id = str(metadata.get("episode_id") or episode_dir.name)
    video_id = str(metadata["video_id"])
    analysis_start, analysis_end, core_start, core_end = episode_interval(metadata)
    expected_count = analysis_end - analysis_start

    raw_clip = find_raw_clip(episode_dir)
    clip_info = inspect_clip(raw_clip)
    if int(clip_info["frame_count"]) != expected_count:
        raise ValueError(
            f"Cached clip has {clip_info['frame_count']} frames, but episode metadata "
            f"requires {expected_count}: {raw_clip}"
        )

    track = track_cache.get(video_id)
    if track is None:
        track = VideoTrack(tracks_root, video_id)
        track_cache[video_id] = track
    if (int(clip_info["height"]), int(clip_info["width"])) != (
        track.height,
        track.width,
    ):
        raise ValueError(
            f"Clip shape {(clip_info['height'], clip_info['width'])} does not match "
            f"DAM4SAM masks {(track.height, track.width)}"
        )

    output_dir = output_root / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "sam2_masks_fixed_crop.npz"
    summary_path = output_dir / "subject_dino_summary.json"
    adapter_path = output_dir / "dam4sam_bout_adapter_summary.json"
    if npz_path.is_file() and summary_path.is_file() and adapter_path.is_file() and not force:
        existing = read_json(adapter_path)
        if existing.get("schema_version") == SCHEMA_VERSION:
            return {**existing, "status": "reused"}

    packed, masks, valid = track.slice(analysis_start, analysis_end)
    core_mask = np.zeros(expected_count, dtype=bool)
    core_mask[core_start - analysis_start : core_end - analysis_start] = True
    timestamps = np.arange(expected_count, dtype=np.float64) / track.fps

    write_npz_atomic(
        npz_path,
        masks=masks,
        source_masks_packed=packed,
        source_mask_width=np.asarray(track.width, dtype=np.int64),
        timestamps_sec=timestamps,
        core_mask=core_mask,
        mask_valid=valid,
    )

    source_summary = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "episode": metadata,
        "fps": track.fps,
        "source_cached_video": str(raw_clip),
        "source_episode_dir": str(episode_dir.resolve()),
        "source_dam4sam_track": str((tracks_root / video_id).resolve()),
        "measurements": {},
    }
    summary_path.write_text(
        json.dumps(source_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    core_valid = valid[core_mask]
    adapter_summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "episode_id": episode_id,
        "video_id": video_id,
        "behavior": metadata.get("behavior"),
        "label": metadata.get("label"),
        "analysis_start_frame": analysis_start,
        "analysis_end_frame_exclusive": analysis_end,
        "core_start_frame": core_start,
        "core_end_frame_exclusive": core_end,
        "analysis_frame_count": expected_count,
        "core_frame_count": int(core_mask.sum()),
        "valid_core_mask_count": int(core_valid.sum()),
        "valid_core_mask_fraction": (
            float(core_valid.mean()) if len(core_valid) else 0.0
        ),
        "raw_clip": str(raw_clip),
        "output_npz": str(npz_path),
    }
    adapter_path.write_text(
        json.dumps(adapter_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return adapter_summary


def main(argv    = None)  :
    args = parse_args(argv)
    tracks_root = args.tracks_root.expanduser().resolve()
    episodes_root = args.episodes_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for path in (tracks_root, episodes_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(
        path
        for path in episodes_root.iterdir()
        if path.is_dir() and (path / "episode.json").is_file()
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        episode_dirs = episode_dirs[: args.limit]
    if not episode_dirs:
        raise RuntimeError(f"No episode.json files found under {episodes_root}")

    rows   = []
    failures   = []
    track_cache   = {}
    prepared = 0
    reused = 0

    for index, episode_dir in enumerate(episode_dirs, start=1):
        print(f"[{index}/{len(episode_dirs)}] {episode_dir.name}", flush=True)
        try:
            row = prepare_episode(
                episode_dir,
                output_root,
                tracks_root,
                track_cache,
                force=args.force,
            )
            rows.append(row)
            if row["status"] == "reused":
                reused += 1
            else:
                prepared += 1
        except Exception as error:
            failure = {
                "episode_id": episode_dir.name,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            print(
                f"  FAILED: {failure['error_type']}: {failure['error']}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise

    write_csv(output_root / "dam4sam_bout_inventory.csv", rows)
    write_csv(output_root / "dam4sam_bout_adapter_failures.csv", failures)
    report = {
        "schema_version": SCHEMA_VERSION,
        "n_selected": len(episode_dirs),
        "n_prepared": prepared,
        "n_reused": reused,
        "n_failed": len(failures),
        "n_source_videos_loaded": len(track_cache),
        "tracks_root": str(tracks_root),
        "episodes_root": str(episodes_root),
        "output_root": str(output_root),
    }
    (output_root / "dam4sam_bout_adapter_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
