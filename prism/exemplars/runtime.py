
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from .config import BEHAVIORS, FPS, DURATION, workspace


def probe(results_dir, repeats=3, profile_power=None, child=None):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    root, meta = workspace(results_dir, profile_power)
    power = meta["flow_lag_profile_power"]
    if child is not None:
        from .simulation import render, measure
        row = json.loads(child)
        frames, masks = render(row)
        measure(frames, masks, profile_power=power)
        t = time.perf_counter()
        result = measure(frames, masks, profile_power=power)
        result.update(wall_sec=time.perf_counter()-t, behavior=row["behavior"],
                      duration_sec=DURATION, fps=FPS, frames=len(frames),
                      flow_lag_profile_power=power, numeric_profile_payload_bytes=16,
                      payload_scope="Four float32 values only; excludes identifiers, metadata and uncertainty",
                      exclusions="Oracle masks supplied; no tracking, event detection, rendering, disk IO or cold start")
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_process_rss_mib"] = rss/(1024**2 if sys.platform == "darwin" else 1024)
        except ImportError:
            import psutil
            mem = psutil.Process().memory_info()
            result["peak_process_rss_mib"] = getattr(mem, "peak_wset", mem.rss)/1024**2
            result["rss_note"] = "Current RSS if peak working set unavailable on this platform"
        return result
    import pandas as pd
    rows = []
    for behavior in BEHAVIORS:
        for i in range(repeats):
            row = dict(condition_id="runtime_"+behavior, behavior=behavior, seed=20261017,
                       frequency_input=1., amplitude_scale=1., disruption=0.)
            process = subprocess.run([sys.executable,"-m","prism.exemplars","runtime",
                "--results-dir",str(root),"--child",json.dumps(row)],
                capture_output=True, text=True, check=True)
            result = json.loads(process.stdout.strip().splitlines()[-1])
            result["repeat"] = i
            rows.append(result)
            print(behavior, i, round(result["wall_sec"],3), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(root/"runtime.csv",index=False)
    summary = frame.groupby("behavior")[["flow_sec","extent_sec","temporal_sec","wall_sec","peak_process_rss_mib"]].median()
    summary.to_csv(root/"runtime_summary.csv")
    (root/"runtime_environment.json").write_text(json.dumps({"platform":platform.platform(),
        "python":sys.version,"flow_lag_profile_power":power},indent=2))
    return summary.to_dict()
