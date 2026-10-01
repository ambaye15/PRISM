from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "intensity": "Intensity",
    "extent": "Extent",
    "frequency_hz": "Frequency",
    "periodicity": "Periodicity",
}


def plot_results(frame, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    colors = {"single_view_2d": "#D55E00", "triangulated_3d": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    for axis, measure in zip(axes.ravel(), LABELS):
        error = measure + ("_absolute_error" if measure == "periodicity" else "_relative_error")
        plot = frame[~frame.method.eq("exact_3d")].copy()
        groups = []
        labels = []
        group_colors = []
        for method, count, label in (
            ("single_view_2d", 1, "Single view"),
            ("triangulated_3d", 2, "2 cameras"),
            ("triangulated_3d", 3, "3 cameras"),
            ("triangulated_3d", 4, "4 cameras"),
        ):
            values = plot[(plot.method == method) & (plot.n_cameras == count)][error].dropna()
            groups.append(values.to_numpy())
            labels.append(label)
            group_colors.append(colors[method])
        artists = axis.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(artists["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_title(LABELS[measure])
        axis.set_ylabel("Absolute score error" if measure == "periodicity" else "Relative error")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Single-view projection versus calibrated 3D reconstruction")
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"multiview_3d_comparison.{suffix}", dpi=300)
    plt.close(fig)

    summary_rows = []
    for method, count_group in frame[~frame.method.eq("exact_3d")].groupby(
        ["method", "n_cameras"]
    ):
        row = {"method": method[0], "n_cameras": int(method[1])}
        for measure in LABELS:
            error = measure + ("_absolute_error" if measure == "periodicity" else "_relative_error")
            row["median_" + error] = float(count_group[error].median())
            row["available_" + measure] = int(count_group[measure].notna().sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "multiview_3d_summary.csv", index=False)
    return summary

