#!/usr/bin/env python3
"""Longitudinal PRISM models with session control and study-aware diagnostics."""


import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import theilslopes

MEASURES = {
    "intensity": "prism_intensity",
    "extent": "prism_extent",
    "periodicity": "prism_periodicity",
    "frequency": "topo_mask_frequency_hz",
    "duration": "duration_sec",
}
BEHAVIORS = ["flap", "flap_rock", "rock"]
COLORS = {"P001": "#4477AA", "P002": "#EE6677", "P003": "#228833",
          "P004": "#CCBB44", "P005": "#66CCEE", "P006": "#AA3377"}


def parse_args()  :
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--strict-mask-valid-fraction", type=float, default=0.90)
    p.add_argument("--strict-flow-valid-fraction", type=float, default=0.90)
    p.add_argument("--strict-ransac-success-fraction", type=float, default=0.80)
    p.add_argument("--minimum-bouts-per-session-cell", type=int, default=3)
    return p.parse_args()


def parse_session_date(value )  :
    value = str(value)
    m = re.search(r"-(20\d{2})-(\d{2})-(\d{2})_study2$", value)
    if m:
        year, month, day = map(int, m.groups())
        return pd.Timestamp(year, month, day)
    m = re.search(r"-(\d{2})-(\d{2})-(\d{2})(?:_\d+)?_study1$", value)
    if m:
        month, day, year = map(int, m.groups())
        return pd.Timestamp(2000 + year, month, day)
    return pd.NaT


def truthy(series )  :
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def transform_outcome(values , measure )   :
    x = pd.to_numeric(values, errors="coerce")
    if measure == "periodicity":
        return np.log(x.clip(1e-4, 1 - 1e-4) / (1 - x.clip(1e-4, 1 - 1e-4))), "logit"
    positive = x[x > 0]
    epsilon = float(positive.min() / 2) if len(positive) else 1e-8
    return np.log(x.clip(lower=epsilon)), "log"


def prepare(df , args )  :
    required = {"participant_id", "session_id", "behavior", "study_id", "fps", "start_sec"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    out = df.copy()
    out["participant_session"] = out["participant_id"].astype(str) + "::" + out["session_id"].astype(str)
    out["session_date"] = out["session_id"].map(parse_session_date)
    if out["session_date"].isna().any():
        bad = sorted(out.loc[out["session_date"].isna(), "session_id"].unique())
        raise ValueError("Could not parse session dates: " + ", ".join(map(str, bad)))
    out["bout_time_minutes"] = pd.to_numeric(out["start_sec"], errors="coerce") / 60.0
    out["time_within_session"] = out["bout_time_minutes"] - out.groupby(
        "participant_session")["bout_time_minutes"].transform("mean")
    first_person = out.groupby("participant_id")["session_date"].transform("min")
    first_person_study = out.groupby(["participant_id", "study_id"])["session_date"].transform("min")
    out["days_from_first_session"] = (out["session_date"] - first_person).dt.days
    out["days_within_participant_study"] = (out["session_date"] - first_person_study).dt.days

    strict_topology = pd.Series(True, index=out.index)
    if "topo_mask_valid_fraction" in out:
        strict_topology &= pd.to_numeric(out["topo_mask_valid_fraction"], errors="coerce").ge(
            args.strict_mask_valid_fraction)
    strict_intensity = pd.Series(True, index=out.index)
    if "optical_flow_valid_pair_fraction" in out:
        strict_intensity &= pd.to_numeric(out["optical_flow_valid_pair_fraction"], errors="coerce").ge(
            args.strict_flow_valid_fraction)
    if "optical_flow_ransac_success_fraction" in out:
        strict_intensity &= pd.to_numeric(out["optical_flow_ransac_success_fraction"], errors="coerce").ge(
            args.strict_ransac_success_fraction)
    if "optical_flow_measurement_reliable" in out:
        strict_intensity &= truthy(out["optical_flow_measurement_reliable"])
    out["strict_qc_intensity"] = strict_intensity
    out["strict_qc_topology"] = strict_topology
    out["strict_qc_duration"] = True
    return out


def strict_qc_column(measure )  :
    if measure == "intensity":
        return "strict_qc_intensity"
    if measure in {"extent", "periodicity", "frequency"}:
        return "strict_qc_topology"
    return "strict_qc_duration"


def complete_for_measure(df , measure , column )  :
    z = df.copy()
    if measure == "frequency":
        if "frequency_eligible" not in z:
            raise ValueError("frequency_eligible is required for frequency analysis")
        z = z.loc[truthy(z["frequency_eligible"])]
    z["outcome"], z["outcome_transform"] = transform_outcome(z[column], measure)
    return z.dropna(subset=["outcome", "time_within_session", "participant_id", "session_id", "behavior"])


def extract_slope(fit, term , transform , time_unit )  :
    slope = float(fit.params[term])
    return {
        "slope_transformed_scale": slope,
        "slope_interpretation": (
            100.0 * (np.exp(slope) - 1.0) if transform == "log" else slope
        ),
        "interpretation_units": (
            f"percent_change_per_{time_unit}" if transform == "log"
            else f"log_odds_change_per_{time_unit}"
        ),
    }


def cluster_sample(df , selected )  :
    pieces = []
    for replicate_id, participant in enumerate(selected):
        piece = df.loc[df["participant_id"] == participant].copy()
        piece["bootstrap_participant"] = f"p{replicate_id}"
        piece["bootstrap_session"] = (
            f"p{replicate_id}::" + piece["session_id"].astype(str)
        )
        pieces.append(piece)
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def percentile_interval(draws )    :
    a = np.asarray(draws, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, 0
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), len(a)


def fit_within_session(df , measure , column ,
                       strict , selections )  :
    z = complete_for_measure(df, measure, column)
    if strict:
        z = z.loc[z[strict_qc_column(measure)]]
    transform = str(z["outcome_transform"].iloc[0]) if len(z) else ""
    rows = []
    for behavior in BEHAVIORS:
        q = z.loc[z["behavior"] == behavior].copy()
        if len(q) < 10 or q["session_id"].nunique() < 2:
            continue
        formula = "outcome ~ time_within_session + C(session_id)"
        fit = smf.ols(formula, q).fit()
        point = extract_slope(fit, "time_within_session", transform, "minute")
        draws = []
        for selected in selections:
            sample = cluster_sample(q, selected)
            if sample.empty or sample["bootstrap_session"].nunique() < 2:
                draws.append(np.nan)
                continue
            try:
                bfit = smf.ols("outcome ~ time_within_session + C(bootstrap_session)", sample).fit()
                draws.append(float(bfit.params["time_within_session"]))
            except Exception:
                draws.append(np.nan)
        low, high, successful = percentile_interval(draws)
        row = {"measure": measure, "column": column, "analysis": "within_session",
               "behavior": behavior, "strict_qc": strict, "n_bouts": len(q),
               "n_participants": q["participant_id"].nunique(),
               "n_sessions": q["session_id"].nunique(), "transform": transform,
               **point, "ci_low_transformed_scale": low, "ci_high_transformed_scale": high,
               "successful_bootstraps": successful}
        if transform == "log":
            row["ci_low_interpretable"] = 100 * (np.exp(low) - 1)
            row["ci_high_interpretable"] = 100 * (np.exp(high) - 1)
        else:
            row["ci_low_interpretable"] = low
            row["ci_high_interpretable"] = high
        rows.append(row)
    return rows


def session_summary(df , args )  :
    group_cols = ["participant_id", "session_id", "participant_session", "study_id",
                  "session_date", "days_from_first_session", "days_within_participant_study",
                  "behavior"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for measure, column in MEASURES.items():
            if column not in group:
                continue
            q = group.loc[truthy(group["frequency_eligible"])] if measure == "frequency" else group
            values = pd.to_numeric(q[column], errors="coerce").dropna()
            if len(values) < args.minimum_bouts_per_session_cell:
                continue
            transformed, transform = transform_outcome(values, measure)
            rows.append({**base, "measure": measure, "column": column,
                         "n_bouts": len(values), "median_raw": float(values.median()),
                         "q25_raw": float(values.quantile(.25)), "q75_raw": float(values.quantile(.75)),
                         "median_transformed": float(np.nanmedian(transformed)), "transform": transform,
                         "strict_qc_fraction": float(q[strict_qc_column(measure)].mean())})
    return pd.DataFrame(rows)


def fit_between_sessions(summary , selections )  :
    rows = []
    for measure in summary["measure"].unique():
        for behavior in BEHAVIORS:
            q = summary.loc[(summary["measure"] == measure) & (summary["behavior"] == behavior)].copy()
            if len(q) < 8 or q["participant_id"].nunique() < 3:
                continue
            # Time is reset within participant and study. This separates short-term
            # longitudinal change from the confounded 2008-to-2010/11 study contrast.
            formula = "median_transformed ~ days_within_participant_study + C(participant_id) + C(study_id)"
            try:
                fit = smf.wls(formula, q, weights=np.sqrt(q["n_bouts"])).fit()
            except Exception:
                continue
            transform = str(q["transform"].iloc[0])
            point = extract_slope(fit, "days_within_participant_study", transform, "day")
            draws = []
            participants = np.asarray(sorted(q["participant_id"].unique()))
            for selected in selections:
                selected = np.asarray([p for p in selected if p in participants])
                sample = cluster_sample(q, selected)
                if sample.empty or sample["bootstrap_participant"].nunique() < 2:
                    draws.append(np.nan)
                    continue
                try:
                    bfit = smf.wls(
                        "median_transformed ~ days_within_participant_study + C(bootstrap_participant) + C(study_id)",
                        sample, weights=np.sqrt(sample["n_bouts"])).fit()
                    draws.append(float(bfit.params["days_within_participant_study"]))
                except Exception:
                    draws.append(np.nan)
            low, high, successful = percentile_interval(draws)
            row = {"measure": measure, "analysis": "between_session_within_study",
                   "behavior": behavior, "n_session_cells": len(q),
                   "n_participants": q["participant_id"].nunique(),
                   "n_sessions": q["session_id"].nunique(), "transform": transform,
                   **point, "ci_low_transformed_scale": low, "ci_high_transformed_scale": high,
                   "successful_bootstraps": successful}
            if transform == "log":
                row["ci_low_interpretable"] = 100 * (np.exp(low) - 1)
                row["ci_high_interpretable"] = 100 * (np.exp(high) - 1)
            else:
                row["ci_low_interpretable"] = low
                row["ci_high_interpretable"] = high
            rows.append(row)
    return rows


def theil_sen_slopes(summary )  :
    rows = []
    for (measure, participant, study, behavior), q in summary.groupby(
        ["measure", "participant_id", "study_id", "behavior"]
    ):
        q = q.drop_duplicates("session_id").sort_values("days_within_participant_study")
        if q["session_id"].nunique() < 3 or q["days_within_participant_study"].nunique() < 3:
            continue
        slope, intercept, low, high = theilslopes(
            q["median_transformed"], q["days_within_participant_study"], .95)
        rows.append({"measure": measure, "participant_id": participant, "study_id": study,
                     "behavior": behavior, "n_sessions": q["session_id"].nunique(),
                     "transform": q["transform"].iloc[0], "slope_transformed_per_day": slope,
                     "ci_low": low, "ci_high": high, "intercept": intercept})
    return pd.DataFrame(rows)


def study_contrasts(summary )  :
    rows = []
    for measure in summary["measure"].unique():
        q = summary.loc[summary["measure"] == measure].copy()
        if q["study_id"].nunique() < 2:
            continue
        try:
            fit = smf.wls(
                "median_transformed ~ C(study_id) + C(behavior) + C(participant_id)",
                q, weights=np.sqrt(q["n_bouts"])).fit()
            term = next((t for t in fit.params.index if t.startswith("C(study_id)")), None)
            if term:
                rows.append({"measure": measure, "term": term, "estimate": fit.params[term],
                             "standard_error_naive": fit.bse[term], "n_session_cells": len(q),
                             "interpretation": "Combined study/setting/calendar-period contrast; not a pure room effect."})
        except Exception as exc:
            rows.append({"measure": measure, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def identifiability_report(df )  :
    sessions = df[["participant_id", "session_id", "study_id", "session_date"]].drop_duplicates()
    rows = []
    for participant, q in sessions.groupby("participant_id"):
        row = {"participant_id": participant, "n_sessions": len(q),
               "first_date": q["session_date"].min(), "last_date": q["session_date"].max(),
               "span_days": (q["session_date"].max() - q["session_date"].min()).days}
        for study in ["study1", "study2"]:
            z = q.loc[q["study_id"] == study]
            row[f"n_sessions_{study}"] = len(z)
            row[f"span_days_{study}"] = (
                (z["session_date"].max() - z["session_date"].min()).days if len(z) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main()  :
    args = parse_args()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_csv(args.input.expanduser().resolve()), args)
    measures = {k: v for k, v in MEASURES.items() if v in df.columns}
    participants = np.asarray(sorted(df["participant_id"].dropna().unique()))
    rng = np.random.default_rng(args.seed)
    selections = rng.choice(participants, size=(args.bootstrap_replicates, len(participants)), replace=True)

    within_rows = []
    for measure, column in measures.items():
        within_rows.extend(fit_within_session(df, measure, column, False, selections))
        within_rows.extend(fit_within_session(df, measure, column, True, selections))
    within = pd.DataFrame(within_rows)
    sessions = session_summary(df, args)
    between = pd.DataFrame(fit_between_sessions(sessions, selections))
    individual = theil_sen_slopes(sessions)
    contrasts = study_contrasts(sessions)
    identifiability = identifiability_report(df)

    df.to_csv(output / "prism_longitudinal_bout_table.csv", index=False)
    sessions.to_csv(output / "participant_session_behavior_medians.csv", index=False)
    within.to_csv(output / "within_session_time_trends.csv", index=False)
    between.to_csv(output / "between_session_within_study_trends.csv", index=False)
    individual.to_csv(output / "participant_theil_sen_trends.csv", index=False)
    contrasts.to_csv(output / "study_combined_setting_time_contrasts.csv", index=False)
    identifiability.to_csv(output / "longitudinal_identifiability.csv", index=False)
    from .plotting.variability import plot_longitudinal_trajectories

    plot_longitudinal_trajectories(sessions, output, BEHAVIORS, COLORS)

    report = {"n_bouts": len(df), "n_participants": len(participants),
              "n_sessions": int(df["session_id"].nunique()),
              "n_session_behavior_measure_cells": len(sessions),
              "study_time_warning": (
                  "Study 1 (laboratory, 2008) and Study 2 (classroom, 2010-2011) differ in both "
                  "setting and calendar period. Their contrast is not a causal room effect."
              ),
              "unmeasured_confounders": ["room", "camera viewpoint/setup", "elicitation context",
                                          "behavioral/physiological state"],
              "primary_analysis": "Within-session trends with session fixed effects",
              "between_session_analysis": "Time reset within participant-study; participant and study adjusted",
              "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed}
    (output / "longitudinal_analysis_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nWITHIN-SESSION TRENDS\n" + within.to_string(index=False))
    print("\nBETWEEN-SESSION WITHIN-STUDY TRENDS\n" + between.to_string(index=False))
    print(f"\nOutputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
