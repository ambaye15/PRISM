#!/usr/bin/env python3
"""Participant variability, mixed-effects ICCs, and frequency QC for PRISM."""


import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MEASURES = {
    "intensity": "prism_intensity",
    "extent": "prism_extent",
    "periodicity": "prism_periodicity",
    "frequency": "prism_frequency_hz",
    "duration": "duration_sec",
}
BEHAVIOR_ORDER = ["flap", "flap_rock", "rock"]
BEHAVIOR_LABELS = {"flap": "Flapping", "flap_rock": "Flapping--rocking", "rock": "Rocking"}
COLORS = {"flap": "#4477AA", "flap_rock": "#EE7733", "rock": "#228833"}


def parse_args()  :
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Final bout-level PRISM CSV (analysis-ready or merged validation table).")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--min-cycles", type=float, default=2.5)
    p.add_argument("--minimum-mask-valid-fraction", type=float, default=0.80)
    p.add_argument("--nyquist-fraction", type=float, default=0.80,
                   help="Require frequency <= this fraction of the Nyquist frequency.")
    p.add_argument("--minimum-cell-size", type=int, default=3,
                   help="Minimum bouts in a participant-by-behavior summary cell.")
    return p.parse_args()


def numeric_column(df , name )  :
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def normalize_columns(df )  :
    df = df.copy()
    required = [
        "participant_id",
        "behavior",
        "session_id",
        "prism_intensity",
        "prism_extent",
        "prism_periodicity",
        "prism_frequency_hz",
        "duration_sec",
    ]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df["behavior"] = df["behavior"].astype(str).str.lower().str.replace("-", "_", regex=False)
    return df


def frequency_eligibility(df , args )  :
    out = df.copy()
    freq = numeric_column(out, "prism_frequency_hz")
    fps = numeric_column(out, "fps")
    cycles = numeric_column(out, "topo_flow_estimated_cycles")
    valid_fraction = numeric_column(out, "topo_flow_valid_mask_fraction")
    reliable = out.get("topo_flow_period_reliable", pd.Series(False, index=out.index))
    reliable = reliable.fillna(False).astype(bool)
    finite_positive = freq.notna() & freq.gt(0)
    cycle_ok = cycles.notna() & cycles.ge(args.min_cycles)
    mask_ok = valid_fraction.notna() & valid_fraction.ge(args.minimum_mask_valid_fraction)
    nyquist_hz = fps / 2.0
    nyquist_ok = fps.notna() & freq.le(args.nyquist_fraction * nyquist_hz)
    out["frequency_nyquist_hz"] = nyquist_hz
    out["frequency_nyquist_ratio"] = freq / nyquist_hz
    out["frequency_eligible"] = reliable & finite_positive & cycle_ok & mask_ok & nyquist_ok
    reasons = np.full(len(out), "eligible", dtype=object)
    reasons[~finite_positive.to_numpy()] = "missing_or_nonpositive_frequency"
    reasons[(finite_positive & ~reliable).to_numpy()] = "period_estimator_unreliable"
    reasons[(finite_positive & reliable & ~cycle_ok).to_numpy()] = "fewer_than_minimum_cycles"
    reasons[(finite_positive & reliable & cycle_ok & ~mask_ok).to_numpy()] = "insufficient_mask_coverage"
    reasons[(finite_positive & reliable & cycle_ok & mask_ok & ~nyquist_ok).to_numpy()] = "too_close_to_nyquist"
    out["frequency_exclusion_reason"] = reasons
    return out


def cell_summary(df , measures  , minimum_n )  :
    rows  = []
    for (participant, session, behavior), group in df.groupby(
        ["participant_id", "session_id", "behavior"], dropna=False
    ):
        for label, column in measures.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if label == "frequency":
                values = pd.to_numeric(group.loc[group["frequency_eligible"], column], errors="coerce").dropna()
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            sd = float(values.std(ddof=1)) if n >= 2 else np.nan
            rows.append({
                "participant_id": participant,
                "session_id": session,
                "behavior": behavior,
                "measure": label,
                "column": column,
                "n": n,
                "median": float(values.median()) if n else np.nan,
                "q25": float(values.quantile(0.25)) if n else np.nan,
                "q75": float(values.quantile(0.75)) if n else np.nan,
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if n else np.nan,
                "mean": mean,
                "sd": sd,
                "coefficient_of_variation": sd / abs(mean) if n >= minimum_n and abs(mean) > 1e-12 else np.nan,
                "eligible_for_variability_summary": n >= minimum_n,
            })
    return pd.DataFrame(rows)


def participant_behavior_summary(df , measures  , minimum_n )  :
    rows  = []
    for (participant, behavior), group in df.groupby(["participant_id", "behavior"], dropna=False):
        for label, column in measures.items():
            subset = group.loc[group["frequency_eligible"]] if label == "frequency" else group
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            sd = float(values.std(ddof=1)) if n >= 2 else np.nan
            rows.append({
                "participant_id": participant,
                "behavior": behavior,
                "measure": label,
                "column": column,
                "n": n,
                "median": float(values.median()) if n else np.nan,
                "q25": float(values.quantile(0.25)) if n else np.nan,
                "q75": float(values.quantile(0.75)) if n else np.nan,
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if n else np.nan,
                "mean": mean,
                "sd": sd,
                "coefficient_of_variation": sd / abs(mean) if n >= minimum_n and abs(mean) > 1e-12 else np.nan,
                "eligible_for_variability_summary": n >= minimum_n,
            })
    return pd.DataFrame(rows)


def aggregate_variability(pb )  :
    valid = pb.loc[pb["eligible_for_variability_summary"]].copy()
    rows = []
    for (measure, behavior), group in valid.groupby(["measure", "behavior"]):
        rows.append({
            "measure": measure,
            "behavior": behavior,
            "n_participants": group["participant_id"].nunique(),
            "median_within_participant_iqr": group["iqr"].median(),
            "q25_within_participant_iqr": group["iqr"].quantile(0.25),
            "q75_within_participant_iqr": group["iqr"].quantile(0.75),
            "median_within_participant_cv": group["coefficient_of_variation"].median(),
            "minimum_participant_median": group["median"].min(),
            "maximum_participant_median": group["median"].max(),
            "range_of_participant_medians": group["median"].max() - group["median"].min(),
        })
    return pd.DataFrame(rows)


def bootstrap_variability(pb , replicates , seed )  :
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(pb["participant_id"].dropna().unique()))
    rows = []
    for (measure, behavior), group in pb.loc[pb["eligible_for_variability_summary"]].groupby(["measure", "behavior"]):
        available = np.asarray(sorted(group["participant_id"].dropna().unique()))
        if not len(available):
            continue
        statistics = {"median_within_participant_iqr": [], "median_within_participant_cv": [],
                      "range_of_participant_medians": []}
        for _ in range(replicates):
            selected = rng.choice(participants, size=len(participants), replace=True)
            sample = pd.concat([group.loc[group["participant_id"] == p] for p in selected], ignore_index=True)
            if sample.empty:
                continue
            statistics["median_within_participant_iqr"].append(sample["iqr"].median())
            statistics["median_within_participant_cv"].append(sample["coefficient_of_variation"].median())
            statistics["range_of_participant_medians"].append(sample["median"].max() - sample["median"].min())
        for statistic, values in statistics.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            rows.append({
                "measure": measure,
                "behavior": behavior,
                "statistic": statistic,
                "estimate": ({
                    "median_within_participant_iqr": group["iqr"].median(),
                    "median_within_participant_cv": group["coefficient_of_variation"].median(),
                    "range_of_participant_medians": group["median"].max() - group["median"].min(),
                })[statistic],
                "ci_low": np.percentile(values, 2.5) if len(values) else np.nan,
                "ci_high": np.percentile(values, 97.5) if len(values) else np.nan,
                "successful_bootstraps": len(values),
            })
    return pd.DataFrame(rows)


def mixed_effects_icc(df , measures  )  :
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("statsmodels is required: python -m pip install statsmodels") from exc
    rows = []
    for label, column in measures.items():
        subset = df.loc[df["frequency_eligible"]].copy() if label == "frequency" else df.copy()
        subset = subset[[column, "behavior", "participant_id", "session_id"]].dropna()
        subset = subset.rename(columns={column: "value"})
        result = {"measure": label, "column": column, "n": len(subset),
                  "n_participants": subset["participant_id"].nunique(),
                  "n_sessions": subset["session_id"].nunique()}
        if len(subset) < 20 or subset["participant_id"].nunique() < 3:
            result.update({"model_converged": False, "error": "insufficient_data"})
            rows.append(result)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    "value ~ C(behavior)", subset,
                    groups=subset["participant_id"],
                    re_formula="1",
                    vc_formula={"session": "0 + C(session_id)"},
                ).fit(reml=True, method="lbfgs", maxiter=1000, disp=False)
            participant_var = float(fit.cov_re.iloc[0, 0])
            session_var = float(fit.vcomp[0]) if len(fit.vcomp) else 0.0
            residual_var = float(fit.scale)
            total = participant_var + session_var + residual_var
            result.update({
                "model_converged": bool(fit.converged),
                "participant_variance": participant_var,
                "session_variance": session_var,
                "residual_within_session_variance": residual_var,
                "participant_icc": participant_var / total if total > 0 else np.nan,
                "session_icc": session_var / total if total > 0 else np.nan,
                "participant_plus_session_icc": (participant_var + session_var) / total if total > 0 else np.nan,
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "error": "",
            })
        except Exception as exc:
            result.update({"model_converged": False, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(result)
    return pd.DataFrame(rows)


def behavior_summary(df , measures  )  :
    rows = []
    for behavior in BEHAVIOR_ORDER:
        group = df.loc[df["behavior"] == behavior]
        for label, column in measures.items():
            subset = group.loc[group["frequency_eligible"]] if label == "frequency" else group
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            rows.append({"behavior": behavior, "measure": label, "n": len(values),
                         "median": values.median(), "q25": values.quantile(0.25),
                         "q75": values.quantile(0.75), "mean": values.mean(), "sd": values.std(ddof=1)})
    return pd.DataFrame(rows)


def frequency_qc_summary(df )  :
    return (df.groupby(["behavior", "frequency_exclusion_reason"], dropna=False)
              .size().rename("n").reset_index())


def main()  :
    args = parse_args()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    df = normalize_columns(pd.read_csv(args.input.expanduser().resolve()))
    df = frequency_eligibility(df, args)
    measures = {label: column for label, column in DEFAULT_MEASURES.items() if column in df.columns}
    if not measures:
        raise ValueError("No recognized PRISM measurement columns were found")

    cells = cell_summary(df, measures, args.minimum_cell_size)
    pb = participant_behavior_summary(df, measures, args.minimum_cell_size)
    aggregate = aggregate_variability(pb)
    bootstrap = bootstrap_variability(pb, args.bootstrap_replicates, args.seed)
    icc = mixed_effects_icc(df, measures)
    behaviors = behavior_summary(df, measures)
    frequency_qc = frequency_qc_summary(df)

    df.to_csv(output / "prism_bouts_with_frequency_eligibility.csv", index=False)
    cells.to_csv(output / "session_participant_behavior_summaries.csv", index=False)
    pb.to_csv(output / "participant_behavior_summaries.csv", index=False)
    aggregate.to_csv(output / "variability_summary.csv", index=False)
    bootstrap.to_csv(output / "participant_clustered_variability_intervals.csv", index=False)
    icc.to_csv(output / "mixed_effects_variance_icc.csv", index=False)
    behaviors.to_csv(output / "behavior_measurement_summaries.csv", index=False)
    frequency_qc.to_csv(output / "frequency_eligibility_summary.csv", index=False)
    from .plotting.variability import plot_behavior_distributions

    plot_behavior_distributions(
        df, measures, output, BEHAVIOR_ORDER, BEHAVIOR_LABELS, COLORS
    )

    report = {
        "n_bouts": len(df), "n_participants": int(df["participant_id"].nunique()),
        "n_sessions": int(df["session_id"].nunique()), "measures": measures,
        "frequency_eligible": int(df["frequency_eligible"].sum()),
        "frequency_total": int(numeric_column(df, "prism_frequency_hz").notna().sum()),
        "settings": {"minimum_cell_size": args.minimum_cell_size, "min_cycles": args.min_cycles,
                     "minimum_mask_valid_fraction": args.minimum_mask_valid_fraction,
                     "nyquist_fraction": args.nyquist_fraction,
                     "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed},
    }
    (output / "variability_analysis_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nBEHAVIOR SUMMARIES\n" + behaviors.to_string(index=False))
    print("\nMIXED-EFFECTS ICC\n" + icc.to_string(index=False))
    print(f"\nOutputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
