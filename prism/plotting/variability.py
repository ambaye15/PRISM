import numpy as np
import pandas as pd


def plot_behavior_distributions(
    data,
    measures,
    output,
    behavior_order,
    behavior_labels,
    colors,
):
    import matplotlib.pyplot as plt

    names = list(measures.items())
    ncols = 2
    nrows = int(np.ceil(len(names) / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(12, 4.1 * nrows), squeeze=False
    )
    random = np.random.default_rng(4401)
    participants = sorted(data["participant_id"].dropna().unique())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<"]
    for axis, (label, column) in zip(axes.flat, names):
        for index, behavior in enumerate(behavior_order, start=1):
            subset = data.loc[data["behavior"] == behavior]
            if label == "frequency":
                subset = subset.loc[subset["frequency_eligible"]]
            values = pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy()
            if len(values):
                parts = axis.violinplot(
                    [values],
                    positions=[index],
                    widths=0.72,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(colors[behavior])
                    body.set_alpha(0.30)
                parts["cmedians"].set_color(colors[behavior])
            for participant_index, participant in enumerate(participants):
                participant_values = subset.loc[
                    subset["participant_id"] == participant, column
                ]
                participant_values = pd.to_numeric(
                    participant_values, errors="coerce"
                ).dropna()
                if len(participant_values):
                    axis.scatter(
                        index + random.normal(0, 0.025),
                        participant_values.median(),
                        marker=markers[participant_index % len(markers)],
                        s=42,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=4,
                    )
        axis.set_xticks(
            range(1, len(behavior_order) + 1),
            [behavior_labels[item] for item in behavior_order],
            rotation=12,
        )
        axis.set_ylabel(label.replace("_", " ").title())
        axis.grid(axis="y", color="0.91")
        axis.set_title(f"{label.replace('_', ' ').title()} by behavior")
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    figure.suptitle(
        "PRISM bout distributions; symbols show participant medians", y=1.01
    )
    figure.tight_layout()
    figure.savefig(
        output / "participant_behavior_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output / "participant_behavior_distributions.pdf", bbox_inches="tight"
    )
    plt.close(figure)


def plot_longitudinal_trajectories(summary, output, behaviors, colors):
    import matplotlib.pyplot as plt

    for measure in summary["measure"].unique():
        selected = summary.loc[summary["measure"] == measure]
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for axis, behavior in zip(axes, behaviors):
            behavior_data = selected.loc[selected["behavior"] == behavior]
            for participant, values in behavior_data.groupby("participant_id"):
                values = values.sort_values("session_date")
                color = colors.get(str(participant), "0.4")
                axis.plot(
                    values["session_date"],
                    values["median_raw"],
                    marker="o",
                    linewidth=1.2,
                    color=color,
                    label=participant,
                )
                for _, row in values.iterrows():
                    marker = "s" if row["study_id"] == "study1" else "o"
                    axis.scatter(
                        row["session_date"],
                        row["median_raw"],
                        marker=marker,
                        color=color,
                        s=35,
                        zorder=3,
                    )
            axis.set_title(behavior.replace("_", "-").title())
            axis.set_xlabel("Recording date")
            axis.tick_params(axis="x", rotation=30)
            axis.grid(color="0.92")
        axes[0].set_ylabel(measure.replace("_", " ").title())
        handles, labels = axes[-1].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[-1].legend(
            unique.values(),
            unique.keys(),
            frameon=False,
            fontsize=8,
            title="Participant",
        )
        figure.suptitle(
            f"{measure.replace('_', ' ').title()} session medians\n"
            "squares: laboratory study; circles: classroom study"
        )
        figure.tight_layout()
        figure.savefig(
            output / f"longitudinal_{measure}.png", dpi=300, bbox_inches="tight"
        )
        figure.savefig(
            output / f"longitudinal_{measure}.pdf", bbox_inches="tight"
        )
        plt.close(figure)

