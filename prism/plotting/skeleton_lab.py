#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

MEASURES = {
    "intensity": ("intensity_truth_height_s", "prism_intensity"),
    "extent": ("extent_truth_deg", "prism_extent"),
    "flow_frequency": ("frequency_truth_hz", "prism_frequency_hz"),
    "welch_frequency": ("frequency_truth_hz", "prism_frequency_welch_hz"),
    "lagged_ssm_frequency": ("frequency_truth_hz", "prism_frequency_lagged_ssm_hz"),
}

DISPLAY = {
    "intensity": ("PRISM intensity", "Optical-flow intensity (body heights/s)"),
    "extent": ("PRISM extent", "Sliding-window orbit radius"),
    "flow_frequency": ("Signed-flow lagged-SSM frequency", "Estimated frequency (Hz)"),
    "welch_frequency": ("PC1--3 Welch frequency", "Estimated frequency (Hz)"),
    "lagged_ssm_frequency": ("Lagged-SSM frequency", "Estimated frequency (Hz)"),
}

ROBUST_DISPLAY = {
    "prism_intensity": "PRISM intensity",
    "prism_extent": "PRISM extent",
    "prism_frequency_hz": "Signed-flow frequency",
    "prism_frequency_welch_hz": "PC1--3 Welch frequency",
    "prism_frequency_lagged_ssm_hz": "Lagged-SSM frequency",
    "prism_periodicity": "SW1PerS periodicity",
}


def _save(fig , stem )  :
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_construct_recovery(clean , out )  :
    """One small-multiple construct-recovery figure for every measurement."""
    behaviors = list(clean["behavior"].drop_duplicates())
    colors = plt.get_cmap("tab10").colors
    for measure, (truth, estimate) in MEASURES.items():
        fig, axes = plt.subplots(1, len(behaviors), figsize=(3.15 * len(behaviors), 3.2),
                                 sharey=False)
        axes = np.atleast_1d(axes)
        for index, (ax, behavior) in enumerate(zip(axes, behaviors)):
            subset = clean.loc[clean.behavior == behavior, [truth, estimate]].replace(
                [np.inf, -np.inf], np.nan).dropna()
            ax.scatter(subset[truth], subset[estimate], s=13, alpha=.42,
                       color=colors[index % len(colors)], linewidths=0)
            if len(subset) > 2 and subset[truth].nunique() > 1 and subset[estimate].nunique() > 1:
                rho = spearmanr(subset[truth], subset[estimate]).statistic
                # A monotone-free linear guide is descriptive only; rho carries the inference.
                slope, intercept = np.polyfit(subset[truth], subset[estimate], 1)
                xx = np.linspace(subset[truth].min(), subset[truth].max(), 100)
                ax.plot(xx, intercept + slope * xx, color="black", lw=1.4)
                annotation = rf"$\rho_s={rho:.2f}$"
            else:
                annotation = ("constant ground truth" if subset[truth].nunique() <= 1
                              else "constant estimate")
            if "frequency" in measure and len(subset):
                lo = min(subset[truth].min(), subset[estimate].min())
                hi = max(subset[truth].max(), subset[estimate].max())
                ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="0.4")
            ax.text(.04, .96, annotation, transform=ax.transAxes, va="top", fontsize=9)
            ax.set_title(behavior.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Controlled ground truth")
            if index == 0:
                ax.set_ylabel(DISPLAY[measure][1])
            ax.grid(alpha=.2)
        fig.suptitle(f"{DISPLAY[measure][0]}: recovery of the controlled construct", y=1.02)
        _save(fig, out / f"construct_recovery_{measure}")


def plot_frequency_duration(duration , out )  :
    """Duration sensitivity for both frequency estimators."""
    pooled = duration.groupby(["method", "duration_sec"], as_index=False).agg(
        median_relative_error=("median_relative_error", "median"),
        within_20_percent=("within_20_percent", "mean"),
    )
    styles = {"PC1-3 Welch": ("o", "#0072B2"), "Lagged SSM": ("s", "#D55E00")}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for method, group in pooled.groupby("method"):
        marker, color = styles.get(method, ("o", None))
        axes[0].plot(group.duration_sec, group.median_relative_error, marker=marker,
                     color=color, lw=2, label=method)
        axes[1].plot(group.duration_sec, group.within_20_percent, marker=marker,
                     color=color, lw=2, label=method)
    axes[0].set_ylabel("Median relative frequency error")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("Estimates within 20% of truth")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("Bout duration (s)")
        ax.set_xticks(sorted(pooled.duration_sec.unique()))
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("Frequency recovery improves with bout duration")
    _save(fig, out / "frequency_accuracy_by_duration")


def plot_individual_robustness(summary , out )  :
    """One uncapped robustness profile per measurement, with failures explicit."""
    for measure, title in ROBUST_DISPLAY.items():
        subset = summary.loc[summary.measure == measure].copy()
        subset["label"] = [f"{a}: {b:g}" for a, b in zip(subset.nuisance, subset.level)]
        subset = subset.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(subset))
        median = subset.median_relative_deviation.to_numpy(float)
        q90 = subset.q90_relative_deviation.to_numpy(float)
        valid = np.isfinite(median)
        # log1p retains zero while allowing errors above 100% to remain visible.
        x_med = np.log1p(median[valid])
        x_q90 = np.log1p(q90[valid])
        fig, ax = plt.subplots(figsize=(8.7, max(7, .27 * len(subset))))
        ax.hlines(y[valid], x_med, x_q90, color="0.65", lw=1.2)
        ax.scatter(x_q90, y[valid], marker="|", color="0.45", s=45,
                   label="90th percentile")
        ax.scatter(x_med, y[valid], color="#0072B2", s=23, zorder=3, label="Median")
        if (~valid).any():
            ax.scatter(np.zeros((~valid).sum()), y[~valid], marker="x", color="#CC3311",
                       s=35, label="Unavailable")
        ax.axvline(np.log1p(.2), color="#CC3311", ls="--", lw=1,
                   label="20% deviation")
        tick_values = np.array([0, .2, 1, 5, 10, 50])
        ax.set_xticks(np.log1p(tick_values),
                      [f"{100*v:g}%" for v in tick_values])
        ax.set_yticks(y, subset.label)
        ax.set_xlabel("Relative deviation from matched clean motion (log scale)")
        ax.set_title(f"Robustness of {title}")
        ax.grid(axis="x", alpha=.22)
        ax.legend(frameon=False, loc="lower right")
        _save(fig, out / f"robustness_{measure}")


def main()  :
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--camera-corrected",
        type=Path,
        default=None,
        help=("Optional corrected clean/camera-shake result table. Rows are matched "
              "by condition_id and replace the corresponding uncorrected rows."),
    )
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    replacement_count = 0
    if args.camera_corrected is not None:
        corrected = pd.read_csv(args.camera_corrected)
        required = {
            "condition_id", "family", "nuisance", "match_id", "behavior", "level",
            "prism_intensity", "prism_extent", "prism_periodicity",
            "prism_frequency_hz", "prism_frequency_welch_hz", "prism_frequency_lagged_ssm_hz",
        }
        missing = sorted(required.difference(corrected.columns))
        if missing:
            raise ValueError(f"Camera-corrected table is missing columns: {missing}")
        corrected = corrected.loc[
            corrected.family.eq("robustness")
            & corrected.nuisance.isin(["clean", "camera_shake_sigma_px"])
        ].copy()
        if corrected.condition_id.duplicated().any():
            duplicates = corrected.loc[corrected.condition_id.duplicated(), "condition_id"].tolist()
            raise ValueError(f"Duplicate corrected condition_id values: {duplicates[:5]}")
        original_ids = set(data.condition_id)
        unknown = sorted(set(corrected.condition_id).difference(original_ids))
        if unknown:
            raise ValueError(
                f"Corrected rows do not match the benchmark manifest; first IDs: {unknown[:5]}"
            )
        replacement_count = len(corrected)
        data = pd.concat(
            [data.loc[~data.condition_id.isin(corrected.condition_id)], corrected],
            ignore_index=True,
            sort=False,
        )
        if data.condition_id.duplicated().any():
            raise ValueError("condition_id is not unique after camera-corrected replacement")
        metadata = {
            "camera_corrected_source": str(args.camera_corrected),
            "replacement_rows": replacement_count,
            "correction_modes": sorted(corrected.get("camera_correction", pd.Series(dtype=str))
                                       .dropna().astype(str).unique().tolist()),
        }
        (args.out / "summary_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    clean = data[data.family == "construct"]
    rows = []
    for behavior, group in clean.groupby("behavior"):
        for measure, (truth, estimate) in MEASURES.items():
            subset = group[[truth, estimate]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = spearmanr(subset[truth], subset[estimate]).statistic if len(subset) > 2 else np.nan
            rel = np.abs(subset[estimate] - subset[truth]) / np.maximum(np.abs(subset[truth]), 1e-12)
            rows.append({"behavior": behavior, "measure": measure, "n": len(subset),
                         "spearman_rho": rho,
                         "median_relative_error": rel.median() if "frequency" in measure else np.nan,
                         "within_20_percent": (rel <= .2).mean() if "frequency" in measure else np.nan})
    construct_summary = pd.DataFrame(rows)
    construct_summary.to_csv(args.out / "construct_validity_summary.csv", index=False)

    # Frequency performance as a function of bout length and observed cycles.
    duration_rows = []
    for method, column in (("Signed-flow lagged SSM", "prism_frequency_hz"),
                           ("PC1-3 Welch", "prism_frequency_welch_hz"),
                           ("Lagged SSM", "prism_frequency_lagged_ssm_hz")):
        subset = clean.copy()
        subset["relative_error"] = (
            (subset[column] - subset["frequency_truth_hz"]).abs()
            / subset["frequency_truth_hz"].abs().clip(lower=1e-12)
        )
        for (behavior, duration), group in subset.groupby(["behavior", "duration_sec"]):
            valid = group["relative_error"].dropna()
            duration_rows.append({"behavior": behavior, "method": method,
                "duration_sec": duration, "n": len(valid),
                "median_true_cycles": group["true_cycles"].median(),
                "median_relative_error": valid.median(),
                "within_10_percent": (valid <= .10).mean() if len(valid) else np.nan,
                "within_20_percent": (valid <= .20).mean() if len(valid) else np.nan})
    duration_summary = pd.DataFrame(duration_rows)
    duration_summary.to_csv(args.out / "frequency_by_duration.csv", index=False)
    plot_construct_recovery(clean, args.out)
    plot_frequency_duration(duration_summary, args.out)

    robust = data[data.family == "robustness"].copy()
    keys = ["match_id"]
    baseline = robust[robust.nuisance == "clean"].set_index(keys)
    deviation_rows = []
    for _, row in robust[robust.nuisance != "clean"].iterrows():
        key = row.match_id
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        for measure in ["prism_intensity", "prism_extent", "prism_frequency_hz", "prism_frequency_welch_hz",
                        "prism_frequency_lagged_ssm_hz", "prism_periodicity"]:
            value, reference = row.get(measure), base.get(measure)
            deviation_rows.append({"behavior": row.behavior, "nuisance": row.nuisance,
                "level": row.level, "seed": row.seed, "measure": measure,
                "relative_deviation": abs(value-reference)/max(abs(reference), 1e-12)
                if pd.notna(value) and pd.notna(reference) else np.nan})
    deviations = pd.DataFrame(deviation_rows)
    deviations.to_csv(args.out / "robustness_deviations.csv", index=False)
    if deviations.empty:
        print("No matched robustness pairs were present; construct summaries were saved.")
        return
    summary = deviations.groupby(["nuisance", "level", "measure"], as_index=False).agg(
        n=("relative_deviation", "count"), median_relative_deviation=("relative_deviation", "median"),
        q90_relative_deviation=("relative_deviation", lambda x: x.quantile(.9)),
        within_20_percent=("relative_deviation", lambda x: (x <= .2).mean()))
    summary.to_csv(args.out / "robustness_summary.csv", index=False)
    plot_individual_robustness(summary, args.out)

    pivot = summary.pivot_table(index=["nuisance", "level"], columns="measure",
                                values="median_relative_deviation")
    fig, ax = plt.subplots(figsize=(11, max(5, .30 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0,
                      vmax=min(1, np.nanquantile(pivot.to_numpy(), .95)))
    ax.set_yticks(range(len(pivot)), [f"{a}: {b:g}" for a, b in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_title("Median relative deviation from matched clean motion")
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(args.out / "robustness_heatmap.png", dpi=220)
    fig.savefig(args.out / "robustness_heatmap.pdf"); plt.close(fig)
    if replacement_count:
        print(f"Used {replacement_count} corrected clean/camera-shake rows")
    print("Saved benchmark summaries to", args.out)


if __name__ == "__main__":
    main()
