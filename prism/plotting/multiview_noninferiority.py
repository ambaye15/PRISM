from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MEASURE_LABELS = {
    "intensity": "Intensity",
    "extent": "Extent",
    "frequency_hz": "Frequency",
    "periodicity": "Periodicity",
}

CATEGORY_LABELS = {
    "clear_single_view": "Clear single view",
    "random_single_view": "Random single view",
    "unfavorable_occluded_view": "Unfavorable + occluded",
    "calibrated_multiview_3d": "Calibrated multiview 3D",
}


def plot_noninferiority(summary, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    margins = (5, 10, 15)
    categories = list(CATEGORY_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True,
                             constrained_layout=True)
    width = 0.18
    x = np.arange(len(MEASURE_LABELS))
    colors = ("#0072B2", "#56B4E9", "#D55E00", "#009E73")
    for axis, margin in zip(axes, margins):
        for offset, (category, color) in enumerate(zip(categories, colors)):
            values = []
            lower = []
            upper = []
            for measure in MEASURE_LABELS:
                selected = summary[
                    summary.category.eq(category) & summary.measure.eq(measure)
                ]
                if len(selected):
                    value = selected[f"noninferior_fraction_{margin}pct"].iloc[0]
                    low = selected[f"noninferior_ci_low_{margin}pct"].iloc[0]
                    high = selected[f"noninferior_ci_high_{margin}pct"].iloc[0]
                    values.append(value)
                    lower.append(value - low)
                    upper.append(high - value)
                else:
                    values.append(np.nan)
                    lower.append(np.nan)
                    upper.append(np.nan)
            axis.bar(
                x + (offset - 1.5) * width,
                values,
                width,
                label=CATEGORY_LABELS[category],
                color=color,
                yerr=np.asarray([lower, upper]),
                capsize=2,
                error_kw={"linewidth": 0.8},
            )
        axis.set_title(f"Margin: {margin}%")
        axis.set_xticks(x)
        axis.set_xticklabels(MEASURE_LABELS.values(), rotation=25, ha="right")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Fraction of conditions non-inferior")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.suptitle("Single-view non-inferiority relative to exact 3D truth", y=1.08)
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"noninferiority_pass_rates.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

