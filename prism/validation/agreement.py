"""Correlation summaries with participant-clustered uncertainty.

The functions in this module are shared by real-video validation baselines.
They operate on paired, finite measurements only and resample participants as
the independent sampling units.
"""

import numpy as np
import pandas as pd
from scipy import stats


def paired_data(data, x, y, behavior=None):
    columns = ["participant_id", "behavior", x, y]
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError("agreement data are missing columns: " + ", ".join(missing))
    paired = data[columns].copy()
    if behavior is not None:
        paired = paired.loc[paired["behavior"] == behavior]
    paired[x] = pd.to_numeric(paired[x], errors="coerce")
    paired[y] = pd.to_numeric(paired[y], errors="coerce")
    finite = np.isfinite(paired[x]) & np.isfinite(paired[y])
    return paired.loc[finite].copy()


def spearman(data, x, y, behavior=None):
    paired = paired_data(data, x, y, behavior)
    if len(paired) < 3 or paired[x].nunique() < 2 or paired[y].nunique() < 2:
        return np.nan, len(paired)
    value = stats.spearmanr(paired[x], paired[y]).statistic
    return float(value), len(paired)


def within_participant_behavior_rank(data, x, y):
    paired = paired_data(data, x, y)
    if len(paired) < 3:
        return np.nan, len(paired)
    groups = ["participant_id", "behavior"]
    paired["x_rank"] = paired.groupby(groups, observed=True)[x].rank(
        method="average", pct=True
    )
    paired["y_rank"] = paired.groupby(groups, observed=True)[y].rank(
        method="average", pct=True
    )
    paired["x_rank"] -= paired.groupby(groups, observed=True)["x_rank"].transform("mean")
    paired["y_rank"] -= paired.groupby(groups, observed=True)["y_rank"].transform("mean")
    if paired["x_rank"].std() == 0 or paired["y_rank"].std() == 0:
        return np.nan, len(paired)
    value = stats.pearsonr(paired["x_rank"], paired["y_rank"]).statistic
    return float(value), len(paired)


def correlation(data, x, y, analysis, behavior=None):
    if analysis == "within_participant_behavior_rank":
        return within_participant_behavior_rank(data, x, y)
    if analysis in {"pooled_spearman", "behavior_spearman"}:
        return spearman(data, x, y, behavior)
    raise ValueError("unknown agreement analysis: " + str(analysis))


def clustered_bootstrap(data, x, y, analysis, behavior, replicates, seed):
    eligible = paired_data(data, x, y, behavior if analysis == "behavior_spearman" else None)
    participants = eligible["participant_id"].dropna().astype(str).unique()
    if len(participants) < 2 or int(replicates) < 1:
        return np.nan, np.nan, 0

    rng = np.random.default_rng(seed)
    values = []
    participant_labels = eligible["participant_id"].astype(str)
    for _ in range(int(replicates)):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        pieces = []
        for copy_index, participant in enumerate(sampled):
            piece = eligible.loc[participant_labels == participant].copy()
            piece["participant_id"] = participant + "__bootstrap_" + str(copy_index)
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        value, _ = correlation(sample, x, y, analysis, behavior)
        if np.isfinite(value):
            values.append(value)

    if not values:
        return np.nan, np.nan, 0
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        len(values),
    )


def agreement_rows(data, x, y, comparison, replicates, seed):
    definitions = [
        ("pooled_spearman", None),
        ("within_participant_behavior_rank", None),
    ]
    definitions.extend(
        ("behavior_spearman", behavior)
        for behavior in sorted(data["behavior"].dropna().unique())
    )

    rows = []
    for index, (analysis, behavior) in enumerate(definitions):
        paired = paired_data(
            data,
            x,
            y,
            behavior if analysis == "behavior_spearman" else None,
        )
        value, n = correlation(data, x, y, analysis, behavior)
        low, high, successful = clustered_bootstrap(
            data, x, y, analysis, behavior, replicates, seed + index
        )
        rows.append({
            "comparison": comparison,
            "x": x,
            "y": y,
            "analysis": analysis,
            "behavior": behavior,
            "n": n,
            "n_participants": int(paired["participant_id"].nunique()),
            "correlation": value,
            "ci_low": low,
            "ci_high": high,
            "successful_bootstraps": successful,
        })
    return rows
