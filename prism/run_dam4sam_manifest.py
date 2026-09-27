#!/usr/bin/env python3
"""Stream a PRISM video through DAM4SAM and retain annotated-bout masks.

The tracker sees every frame from the prompt frame onward. Masks are packed and
saved only where the frame-aligned manifest annotation is nonzero (plus optional
context frames). A full tracking overlay can be written for identity QC.
"""


import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ModuleNotFoundError:  # Allows --help and metadata helpers without OpenCV.
    cv2 = None


LABEL_NAMES = {0: "none", 1: "rock", 2: "flap", 3: "flap_rock"}
SCHEMA_VERSION = "prism.dam4sam.track.v1"


def require_opencv()  :
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is unavailable. Activate dam4sam_env before running tracking."
        )


def parse_args(argv    = None)  :
    parser = argparse.ArgumentParser(
        description=(
            "Track one manifest video continuously with DAM4SAM while retaining "
            "masks only for annotated bouts."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--dam4sam-root", type=Path, default=Path.home() / "DAM4SAM")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root used for relative video paths; defaults to manifest/../..",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "prism_dam4sam_results",
    )
    parser.add_argument(
        "--model",
        default="sam21pp-T",
        choices=("sam21pp-T", "sam21pp-S", "sam21pp-B", "sam21pp-L"),
    )
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Initialization box in pixels. If omitted, draw it interactively.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=0,
        help="Also retain this many frames before and after each annotated bout.",
    )
    parser.add_argument(
        "--write-qc-video",
        action="store_true",
        help="Write a full tracking overlay, including non-bout frames.",
    )
    parser.add_argument(
        "--save-pngs",
        action="store_true",
        help="Also save one PNG for every retained, nonempty mask.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Smoke-test limit measured from the prompt frame.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def load_manifest_entry(manifest , video_id )    :
    raw = manifest.read_bytes()
    data = json.loads(raw)
    matches = [row for row in data.get("videos", []) if row.get("video_id") == video_id]
    if not matches:
        raise KeyError(f"video_id {video_id!r} is absent from {manifest}")
    if len(matches) != 1:
        raise ValueError(f"video_id {video_id!r} occurs {len(matches)} times")
    return matches[0], hashlib.sha256(raw).hexdigest()


def resolve_video_path(
    entry  , manifest , project_root   )   :
    root = (
        project_root.expanduser().resolve()
        if project_root is not None
        else manifest.expanduser().resolve().parent.parent
    )
    source = Path(entry["path"]).expanduser()
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    return source, root


def retained_frames(labels , context_frames )  :
    if context_frames < 0:
        raise ValueError("context_frames must be nonnegative")
    requested = labels != 0
    if context_frames == 0 or not requested.any():
        return requested
    expanded = requested.copy()
    for index in np.flatnonzero(requested):
        lo = max(0, int(index) - context_frames)
        hi = min(len(labels), int(index) + context_frames + 1)
        expanded[lo:hi] = True
    return expanded


def normalize_box(box , width , height )  :
    if len(box) != 4:
        raise ValueError("box must contain x, y, width, height")
    x, y, w, h = [int(round(float(value))) for value in box]
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    w = min(max(w, 1), width - x)
    h = min(max(h, 1), height - y)
    return [x, y, w, h]


def select_box(frame_bgr )  :
    window = "DAM4SAM: draw the target box, then press Enter or Space"
    box = cv2.selectROI(window, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    if box[2] <= 0 or box[3] <= 0:
        raise RuntimeError("No valid initialization box was selected")
    return [int(value) for value in box]


def mask_from_output(output  , height , width )  :
    if "pred_mask" not in output:
        raise KeyError("DAM4SAM output has no 'pred_mask'")
    mask = np.asarray(output["pred_mask"])
    mask = np.squeeze(mask)
    if mask.shape != (height, width):
        raise ValueError(
            f"DAM4SAM mask shape {mask.shape} differs from video shape {(height, width)}"
        )
    return mask.astype(bool, copy=False)


def draw_qc(
    frame_bgr ,
    mask ,
    frame_index ,
    label ,
    retained ,
)  :
    output = frame_bgr.copy()
    colored = output.copy()
    colored[mask] = (0, 210, 0)
    output = cv2.addWeighted(output, 0.55, colored, 0.45, 0.0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 255, 255), 1)
    label_name = LABEL_NAMES.get(int(label), f"label_{label}")
    mode = "retained" if retained else "tracking context"
    text = f"frame {frame_index} | {label_name} | {mode}"
    cv2.putText(
        output,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def import_tracker(dam4sam_root ):
    root = dam4sam_root.expanduser().resolve()
    if not (root / "dam4sam_tracker.py").is_file():
        raise FileNotFoundError(f"Not a DAM4SAM checkout: {root}")
    sys.path.insert(0, str(root))
    from dam4sam_tracker import DAM4SAMTracker

    return DAM4SAMTracker, root


def safe_output_paths(output_dir )   :
    return {
        "packed": output_dir / "dam4sam_video_masks_packed.npy",
        "valid": output_dir / "dam4sam_video_mask_valid.npy",
        "summary": output_dir / "dam4sam_track_summary.json",
        "qc": output_dir / "dam4sam_full_tracking_qc.mp4",
        "packed_partial": output_dir / ".dam4sam_video_masks_packed.partial.npy",
        "valid_partial": output_dir / ".dam4sam_video_mask_valid.partial.npy",
        "png_dir": output_dir / "bout_mask_pngs",
    }


def prepare_outputs(paths  , force , save_pngs )  :
    completed = [paths["packed"], paths["valid"], paths["summary"], paths["qc"]]
    if not force and any(path.exists() for path in completed):
        existing = [str(path) for path in completed if path.exists()]
        raise FileExistsError(
            "Output already exists; use --force to replace this exact video result: "
            + ", ".join(existing)
        )
    for key in ("packed_partial", "valid_partial"):
        if paths[key].exists():
            paths[key].unlink()
    if force:
        for path in completed:
            if path.exists():
                path.unlink()
        if paths["png_dir"].is_dir():
            shutil.rmtree(paths["png_dir"])
    if save_pngs:
        paths["png_dir"].mkdir(parents=True, exist_ok=True)


def run(args )   :
    require_opencv()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    entry, manifest_sha256 = load_manifest_entry(manifest, args.video_id)
    source_video, project_root = resolve_video_path(entry, manifest, args.project_root)

    labels = np.asarray(entry.get("annotations", []), dtype=np.int16)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Manifest annotations must be a nonempty one-dimensional list")
    if np.any(labels < 0):
        raise ValueError("Manifest annotations contain negative labels")
    requested = retained_frames(labels, args.context_frames)
    if not requested.any():
        raise ValueError(f"{args.video_id} has no nonzero annotated frames")
    if args.prompt_frame < 0 or args.prompt_frame >= len(labels):
        raise ValueError("prompt-frame lies outside the manifest annotation array")
    if requested[: args.prompt_frame].any():
        raise ValueError(
            "prompt-frame occurs after requested bout frames; DAM4SAM is forward-only. "
            "Choose an earlier visible prompt frame."
        )
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive")

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {source_video}")
    reported_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = float(entry.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Could not determine a positive video FPS")
    if reported_frames > 0 and reported_frames != len(labels):
        cap.release()
        raise ValueError(
            f"Frame alignment failure: video reports {reported_frames} frames but "
            f"manifest contains {len(labels)} annotations"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.prompt_frame)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        cap.release()
        raise RuntimeError(f"Could not read prompt frame {args.prompt_frame}")
    height, width = frame_bgr.shape[:2]
    box = normalize_box(args.box, width, height) if args.box else select_box(frame_bgr)

    Tracker, dam4sam_root = import_tracker(args.dam4sam_root)
    print(f"Loading {args.model} from {dam4sam_root}", flush=True)
    tracker = Tracker(args.model)

    output_dir = args.output_root.expanduser().resolve() / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = safe_output_paths(output_dir)
    prepare_outputs(paths, args.force, args.save_pngs)

    packed = np.lib.format.open_memmap(
        paths["packed_partial"],
        mode="w+",
        dtype=np.uint8,
        shape=(len(labels), height, (width + 7) // 8),
    )
    valid = np.lib.format.open_memmap(
        paths["valid_partial"], mode="w+", dtype=bool, shape=(len(labels),)
    )
    packed[:] = 0
    valid[:] = False

    writer = None
    if args.write_qc_video:
        writer = cv2.VideoWriter(
            str(paths["qc"]),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create QC video: {paths['qc']}")

    stop_exclusive = len(labels)
    if args.max_frames is not None:
        stop_exclusive = min(stop_exclusive, args.prompt_frame + args.max_frames)

    started_at = time.perf_counter()
    processed = 0
    nonempty_retained = 0
    try:
        frame_index = args.prompt_frame
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        output = tracker.initialize(Image.fromarray(rgb), None, bbox=box)

        while frame_index < stop_exclusive:
            if frame_index != args.prompt_frame:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(
                        f"Video ended at frame {frame_index}; expected {stop_exclusive}"
                    )
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                output = tracker.track(Image.fromarray(rgb))

            mask = mask_from_output(output, height, width)
            keep = bool(requested[frame_index])
            label = int(labels[frame_index])
            if keep and mask.any():
                packed[frame_index] = np.packbits(mask, axis=-1)
                valid[frame_index] = True
                nonempty_retained += 1
                if args.save_pngs:
                    png = paths["png_dir"] / f"frame_{frame_index:07d}_label_{label}.png"
                    if not cv2.imwrite(str(png), mask.astype(np.uint8) * 255):
                        raise RuntimeError(f"Could not write {png}")

            if writer is not None:
                writer.write(draw_qc(frame_bgr, mask, frame_index, label, keep))

            processed += 1
            if processed == 1 or processed % args.progress_every == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                rate = processed / elapsed
                remaining = stop_exclusive - frame_index - 1
                eta_minutes = remaining / max(rate, 1e-9) / 60.0
                print(
                    f"frame={frame_index}/{stop_exclusive - 1} "
                    f"rate={rate:.2f} fps ETA={eta_minutes:.1f} min",
                    flush=True,
                )
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        packed.flush()
        valid.flush()

    del packed
    del valid
    os.replace(paths["packed_partial"], paths["packed"])
    os.replace(paths["valid_partial"], paths["valid"])

    saved_valid = np.load(paths["valid"], mmap_mode="r")
    processed_slice = slice(args.prompt_frame, stop_exclusive)
    requested_processed = requested[processed_slice]
    valid_processed = np.asarray(saved_valid[processed_slice], dtype=bool)
    requested_count = int(requested_processed.sum())
    valid_count = int(valid_processed[requested_processed].sum())
    elapsed = time.perf_counter() - started_at
    counts = Counter(int(value) for value in labels[processed_slice])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "video_id": args.video_id,
        "source_video": str(source_video),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "project_root": str(project_root),
        "frame_count": len(labels),
        "reported_video_frame_count": reported_frames,
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "prompt_frame": args.prompt_frame,
        "box_xywh_pixels": box,
        "context_frames": args.context_frames,
        "processed_start_frame": args.prompt_frame,
        "processed_stop_frame_exclusive": stop_exclusive,
        "processed_frame_count": processed,
        "complete_video_processed": args.prompt_frame == 0 and stop_exclusive == len(labels),
        "requested_processed_frame_count": requested_count,
        "valid_requested_mask_count": valid_count,
        "empty_requested_mask_count": requested_count - valid_count,
        "valid_requested_mask_fraction": valid_count / requested_count,
        "annotation_counts_processed": {
            LABEL_NAMES.get(label, str(label)): count for label, count in sorted(counts.items())
        },
        "elapsed_seconds": elapsed,
        "processing_fps": processed / max(elapsed, 1e-9),
        "outside_requested_rows_are_empty": True,
        "packed_mask_bitorder": "big",
        "outputs": {
            "packed_masks": str(paths["packed"]),
            "mask_validity": str(paths["valid"]),
            "full_tracking_qc_video": str(paths["qc"])
            if args.write_qc_video
            else None,
            "bout_mask_pngs": str(paths["png_dir"]) if args.save_pngs else None,
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main()  :
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted; partial arrays were retained for diagnosis.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
