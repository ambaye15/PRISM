#!/usr/bin/env python3
"""Run the PRISM DAM4SAM manifest tracker sequentially over many videos.

The batch runner selects and records one initialization box per video, invokes
run_dam4sam_manifest.py, skips verified completed results, and safely reruns an
incomplete video after an interruption.
"""


import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BOX_SCHEMA = "prism.dam4sam.boxes.v1"


def parse_args(argv    = None)  :
    parser = argparse.ArgumentParser(
        description="Process every unfinished PRISM manifest video with DAM4SAM."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_dam4sam_manifest.py"),
        help="Path to the tested single-video runner.",
    )
    parser.add_argument("--dam4sam-root", type=Path, default=Path.home() / "DAM4SAM")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "prism_dam4sam_results",
    )
    parser.add_argument(
        "--boxes-file",
        type=Path,
        default=Path("configs/dam4sam_boxes.json"),
        help="Persistent initialization boxes, relative to the current directory.",
    )
    parser.add_argument(
        "--model",
        default="sam21pp-T",
        choices=("sam21pp-T", "sam21pp-S", "sam21pp-B", "sam21pp-L"),
    )
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--write-qc-video", action="store_true")
    parser.add_argument("--save-pngs", action="store_true")
    parser.add_argument(
        "--collect-boxes-only",
        action="store_true",
        help="Select and save boxes for unfinished videos without running tracking.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="VIDEO_ID",
        help="Process only these video IDs, in manifest order.",
    )
    parser.add_argument(
        "--start-at",
        metavar="VIDEO_ID",
        help="Ignore manifest entries before this video ID.",
    )
    parser.add_argument(
        "--reselect-box",
        action="append",
        default=[],
        metavar="VIDEO_ID",
        help="Discard and redraw this video's saved box; may be repeated.",
    )
    parser.add_argument(
        "--rerun",
        action="append",
        default=[],
        metavar="VIDEO_ID",
        help="Regenerate a completed video's outputs; may be repeated.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later videos after a video fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which videos would run or be skipped without opening windows.",
    )
    return parser.parse_args(argv)


def load_manifest(path )   :
    data = json.loads(path.read_text(encoding="utf-8"))
    videos = data.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"Manifest has no nonempty 'videos' list: {path}")
    seen  = set()
    for row in videos:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("Every manifest row must contain a nonempty video_id")
        if video_id in seen:
            raise ValueError(f"Duplicate video_id in manifest: {video_id}")
        seen.add(video_id)
    return videos


def resolve_project_root(manifest , project_root   )  :
    if project_root is not None:
        return project_root.expanduser().resolve()
    return manifest.parent.parent.resolve()


def resolve_video(row  , project_root )  :
    source = Path(row["path"]).expanduser()
    if not source.is_absolute():
        source = project_root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    return source


def load_boxes(path )    :
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != BOX_SCHEMA:
        raise ValueError(f"Unsupported boxes file schema in {path}")
    videos = data.get("videos", {})
    if not isinstance(videos, dict):
        raise ValueError(f"'videos' must be an object in {path}")
    return videos


def save_boxes(path , boxes   )  :
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": BOX_SCHEMA, "videos": boxes}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_box(video_path , frame_index )  :
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is unavailable; activate dam4sam_env") from error

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    window = "Draw child box; press Enter or Space"
    box = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if box[2] <= 0 or box[3] <= 0:
        raise RuntimeError("No valid box was selected")
    return [int(value) for value in box]


def completed_result(output_root , video_id )  :
    directory = output_root / video_id
    summary_path = directory / "dam4sam_track_summary.json"
    packed_path = directory / "dam4sam_video_masks_packed.npy"
    valid_path = directory / "dam4sam_video_mask_valid.npy"
    if not (summary_path.is_file() and packed_path.is_file() and valid_path.is_file()):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (
            summary.get("video_id") == video_id
            and int(summary["processed_stop_frame_exclusive"])
            == int(summary["frame_count"])
            and int(summary["requested_processed_frame_count"]) > 0
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def has_annotated_frames(row  )  :
    annotations = row.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(
            f"Manifest row {row.get('video_id')!r} has no annotation list"
        )
    return any(int(label) != 0 for label in annotations)


def select_rows(
    videos  , only   , start_at   )   :
    ids = [row["video_id"] for row in videos]
    if only:
        unknown = sorted(set(only) - set(ids))
        if unknown:
            raise KeyError(f"Unknown --only video IDs: {', '.join(unknown)}")
        wanted = set(only)
        videos = [row for row in videos if row["video_id"] in wanted]
    if start_at:
        selected_ids = [row["video_id"] for row in videos]
        if start_at not in selected_ids:
            raise KeyError(f"--start-at video ID is not selected: {start_at}")
        videos = videos[selected_ids.index(start_at) :]
    return videos


def append_log(path , message )  :
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{stamp} {message}\n")


def build_command(
    args ,
    manifest ,
    project_root ,
    output_root ,
    video_id ,
    prompt_frame ,
    box ,
    force ,
)  :
    command = [
        sys.executable,
        str(args.runner.expanduser().resolve()),
        "--manifest",
        str(manifest),
        "--video-id",
        video_id,
        "--dam4sam-root",
        str(args.dam4sam_root.expanduser().resolve()),
        "--project-root",
        str(project_root),
        "--output-root",
        str(output_root),
        "--model",
        args.model,
        "--prompt-frame",
        str(prompt_frame),
        "--box",
        *[str(value) for value in box],
        "--context-frames",
        str(args.context_frames),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.write_qc_video:
        command.append("--write-qc-video")
    if args.save_pngs:
        command.append("--save-pngs")
    if force:
        command.append("--force")
    return command


def run_batch(args )  :
    manifest = args.manifest.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not runner.is_file():
        raise FileNotFoundError(f"Single-video runner does not exist: {runner}")
    videos = select_rows(load_manifest(manifest), args.only, args.start_at)
    known_ids = {row["video_id"] for row in videos}
    requested_overrides = set(args.reselect_box) | set(args.rerun)
    unknown_overrides = sorted(requested_overrides - known_ids)
    if unknown_overrides:
        raise KeyError(
            "Override video IDs are not in the selected work list: "
            + ", ".join(unknown_overrides)
        )
    project_root = resolve_project_root(manifest, args.project_root)
    output_root = args.output_root.expanduser().resolve()
    boxes_path = args.boxes_file.expanduser().resolve()
    boxes = load_boxes(boxes_path)
    for video_id in args.reselect_box:
        boxes.pop(video_id, None)
    if args.reselect_box:
        save_boxes(boxes_path, boxes)

    log_path = output_root / "dam4sam_batch.log"
    completed = 0
    failed = 0
    skipped = 0
    boxes_collected = 0

    print(f"Selected {len(videos)} video(s)")
    print(f"Results: {output_root}")
    print(f"Saved boxes: {boxes_path}")

    for number, row in enumerate(videos, start=1):
        video_id = row["video_id"]
        prefix = f"[{number}/{len(videos)}] {video_id}"
        if not has_annotated_frames(row):
            print(f"{prefix}: no annotated frames; skipping", flush=True)
            append_log(log_path, f"SKIP_NO_ANNOTATIONS {video_id}")
            skipped += 1
            continue
        rerun = video_id in requested_overrides
        if completed_result(output_root, video_id) and not rerun:
            print(f"{prefix}: already complete; skipping", flush=True)
            skipped += 1
            continue

        saved = boxes.get(video_id)
        if saved is not None:
            prompt_frame = int(saved["prompt_frame"])
            box = [int(value) for value in saved["box_xywh_pixels"]]
            print(f"{prefix}: reusing saved box {box} at frame {prompt_frame}", flush=True)
            if args.collect_boxes_only:
                skipped += 1
                continue
        else:
            prompt_frame = args.prompt_frame
            if args.dry_run:
                print(f"{prefix}: needs a new box at frame {prompt_frame}", flush=True)
                continue
            video_path = resolve_video(row, project_root)
            print(f"{prefix}: draw a tight box around the child", flush=True)
            box = select_box(video_path, prompt_frame)
            boxes[video_id] = {
                "prompt_frame": prompt_frame,
                "box_xywh_pixels": box,
            }
            save_boxes(boxes_path, boxes)
            boxes_collected += 1
            print(f"{prefix}: saved box {box}", flush=True)
            if args.collect_boxes_only:
                continue

        force = rerun or (output_root / video_id).exists()
        command = build_command(
            args,
            manifest,
            project_root,
            output_root,
            video_id,
            prompt_frame,
            box,
            force,
        )
        if args.dry_run:
            print(f"{prefix}: would run", flush=True)
            continue

        append_log(log_path, f"START {video_id}")
        print(f"{prefix}: tracking started", flush=True)
        try:
            result = subprocess.run(command, check=False)
        except KeyboardInterrupt:
            append_log(log_path, f"INTERRUPTED {video_id}")
            print("\nInterrupted. Run the same command later to resume.", file=sys.stderr)
            return 130

        if result.returncode == 0 and completed_result(output_root, video_id):
            completed += 1
            append_log(log_path, f"COMPLETE {video_id}")
            print(f"{prefix}: complete", flush=True)
        else:
            failed += 1
            append_log(log_path, f"FAILED {video_id} exit={result.returncode}")
            print(f"{prefix}: FAILED (exit {result.returncode})", file=sys.stderr)
            if not args.continue_on_error:
                print("Stopping. Fix the issue, then run the same command again.")
                break

    print(
        f"Batch summary: newly completed={completed}, boxes collected={boxes_collected}, "
        f"skipped={skipped}, failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


def main()  :
    try:
        raise SystemExit(run_batch(parse_args()))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
