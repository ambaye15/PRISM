import numpy as np


def plot_flow_trace(path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    valid = [row for row in rows if row.get("flow_pair_valid")]
    if not valid:
        return
    time = np.asarray([row["timestamp_sec"] for row in valid], dtype=float)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    lines = (
        #("flow_pixel_median_body_heights_per_sec", "pixel median", "#356aa0"),
        #("flow_pixel_p75_body_heights_per_sec", "pixel 75th percentile", "#e07a3f"),
        #("flow_pixel_p90_body_heights_per_sec", "pixel 90th percentile", "#6a9f58"),
        (
            "flow_pixel_p75_torso_corrected_body_heights_per_sec",
            "torso-corrected p75",
            "#8c4b9f",
        ),
    )
    for name, label, color in lines:
        if not any(row.get(name) is not None for row in valid):
            continue
        axis.plot(
            time,
            [row.get(name, np.nan) for row in valid],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=label,
            color=color,
        )
    axis.set(
        title="Subject-masked optical-flow rate",
        xlabel="Cached-clip time (s)",
        ylabel="Body heights per second",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

