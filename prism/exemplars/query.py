import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BEHAVIORS, METRICS, FPS, manifest
from .model import predict_parameters


def nearest(behavior, target, top_k=3, *, results_dir):
    if behavior not in BEHAVIORS:
        raise ValueError(f"Unsupported behavior: {behavior}")
    target = np.asarray(target, float)
    if target.shape != (4,) or np.isinf(target).any():
        raise ValueError("Profile must contain four values; use nan for missing coordinates")
    finite = np.isfinite(target)
    if not finite.any():
        raise ValueError("At least one finite metric is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    root = Path(results_dir)
    manifest(root)
    data = pd.read_csv(root/"library_and_test.csv")
    library = data[(data.split == "library") & (data.status == "ok") &
                   (data.behavior == behavior)].copy()
    if library.empty:
        raise ValueError(f"No available library examples for {behavior}")
    scale = np.asarray(json.loads((root/"scales.json").read_text())[behavior], float)
    if scale.shape != (4,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid library profile scales")
    values = library[METRICS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Usable library contains non-finite metrics")
    library["match_error"] = np.sqrt(np.mean(
        ((values[:, finite]-target[finite])/scale[finite])**2, axis=1))
    return library.sort_values(["match_error", "condition_id"]).head(top_k)


def write_video(frames, path, title, details):
    import cv2
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.font_manager import findfont
    font = ImageFont.truetype(findfont("DejaVu Sans"), 16)
    small = ImageFont.truetype(findfont("DejaVu Sans"), 13)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame in frames:
            canvas = Image.new("RGB", (512, 400), "#f5f7fa")
            draw = ImageDraw.Draw(canvas)
            draw.text((18,13), title, fill="#12263a", font=font)
            for j, line in enumerate(details):
                draw.text((18,40+j*20), line, fill="#334155", font=small)
            im = cv2.cvtColor(cv2.resize(frame, (512,288)), cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(im), (0,110))
            writer.append_data(np.asarray(canvas))


def query(behavior, target, out, *, results_dir, method="both", top_k=3,
          max_distance=.35, no_video=False, seed=7070, profile_power=None):
    if method not in {"retrieval", "neural", "both"}:
        raise ValueError("method must be retrieval, neural or both")
    if max_distance < 0 or not np.isfinite(max_distance):
        raise ValueError("max_distance must be finite and nonnegative")
    root = Path(results_dir)
    metadata = manifest(root)
    power = metadata["flow_lag_profile_power"]
    if profile_power is not None and profile_power != power:
        raise ValueError("Requested profile and model use different lag-distance definitions")
    target = np.asarray(target, float)

    candidates = nearest(behavior, target, top_k, results_dir=root)
    if method in {"neural", "both"} and not np.isfinite(target).all():
        raise ValueError("The neural decoder requires all four finite measurements")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(target=dict(zip(METRICS, [float(x) if np.isfinite(x) else None for x in target])),
                  behavior=behavior, method=method, tolerance=max_distance,
                  flow_lag_profile_power=power, source=metadata["source"],
                  interpretation="Synthetic exemplars; not participant reconstruction or clinical judgment")
    if method in {"retrieval", "both"}:
        report["matches"] = candidates[["condition_id",*METRICS,"match_error"]].to_dict("records")
        report["close_match_found"] = bool(candidates.match_error.iloc[0] <= max_distance)
        if not no_video:
            from .simulation import render
            for j, row in enumerate(candidates.to_dict("records")):
                frames, masks = render(row)
                image_hash = hashlib.sha256(frames.tobytes()+masks.tobytes()).hexdigest()
                if image_hash != row["image_sha256"]:
                    raise ValueError("Rerendered clip differs from measured record; regenerate the library")
                text = "I %.3f | E %.3f | f %.3f Hz | PS %.3f" % tuple(row[k] for k in METRICS)
                write_video(frames, output/f"match_{j+1}.mp4", "Retrieved synthetic "+behavior.replace("_"," "),
                            [text, f'Profile distance {row["match_error"]:.3f} library IQR units',
                             "Illustrative motion; not participant reconstruction"])
    if method in {"neural", "both"}:
        from .simulation import render, measure
        prediction = predict_parameters(behavior, target, results_dir=root)
        row = dict(condition_id="requested_neural_example", behavior=behavior, seed=seed,
                   **prediction["parameters"])
        frames, masks = render(row)
        achieved = measure(frames, masks, profile_power=power)
        observed = np.asarray([achieved[k] for k in METRICS], dtype=float)
        complete = bool(np.isfinite(observed).all())
        scale = np.asarray(json.loads((root/"scales.json").read_text())[behavior])
        error = float(np.sqrt(np.mean(((observed-target)/scale)**2))) if complete else None
        report["neural_example"] = dict(parameters=row, raw_parameters=prediction["raw_parameters"],
            parameter_clipped=prediction["parameter_clipped"],
            measurements={k:float(achieved[k]) if achieved[k] is not None and np.isfinite(achieved[k]) else None for k in METRICS},
            measurement_status="ok" if complete else "unavailable", match_error=error,
            close_match_found=bool(complete and error <= max_distance))
        if not no_video:
            text = "I %.3f | E %.3f | f %.3f Hz | PS %.3f" % tuple(observed)
            distance_text = f"Profile distance {error:.3f} library IQR units" if complete else "Some measurements unavailable"
            write_video(frames, output/"neural_example.mp4", "Neural synthetic "+behavior.replace("_"," "),
                        [text, distance_text, "Illustrative motion; not participant reconstruction"])
    (output/"matches.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
