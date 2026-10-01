from pathlib import Path

import numpy as np


def _plot_diagram(axis, diagram, title):
    if diagram.size:
        persistence = diagram[:, 1] - diagram[:, 0]
        upper = max(float(diagram.max()) * 1.06, 1e-6)
        axis.plot([0, upper], [0, upper], "--", color="0.5", linewidth=1)
        axis.scatter(
            diagram[:, 0],
            diagram[:, 1],
            c=persistence,
            cmap="plasma",
            s=32,
            edgecolor="black",
            linewidth=0.25,
        )
        longest = int(np.argmax(persistence))
        axis.scatter(
            diagram[longest, 0],
            diagram[longest, 1],
            s=100,
            facecolor="none",
            edgecolor="#00bcd4",
            linewidth=1.7,
        )
        axis.set_xlim(0, upper)
        axis.set_ylim(0, upper)
    else:
        axis.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
        axis.text(0.5, 0.35, "No finite H1 classes", ha="center")
    axis.set(title=title, xlabel="Birth", ylabel="Death")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)


def _plot_pca(axis, coordinates, ratios, times, title, retained_positions=None):
    if retained_positions is None:
        retained_positions = np.arange(len(coordinates))
    for index in range(len(coordinates) - 1):
        if retained_positions[index + 1] == retained_positions[index] + 1:
            axis.plot(
                coordinates[index : index + 2, 0],
                coordinates[index : index + 2, 1],
                color="0.75",
                linewidth=0.6,
                zorder=1,
            )
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=times - times[0],
        cmap="viridis",
        s=20,
        edgecolor="none",
        zorder=2,
    )
    axis.scatter(
        coordinates[0, 0],
        coordinates[0, 1],
        s=65,
        marker="o",
        facecolor="none",
        edgecolor="black",
        label="start",
        zorder=3,
    )
    axis.scatter(
        coordinates[-1, 0],
        coordinates[-1, 1],
        s=60,
        marker="X",
        color="black",
        label="end",
        zorder=3,
    )
    pc1 = 100 * ratios[0] if ratios else 0.0
    pc2 = 100 * ratios[1] if len(ratios) > 1 else 0.0
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])

    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return scatter


def plot_topology_diagnostic(result, summary, output, title):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    starts = result["start_times_sec"]
    retained = np.flatnonzero(result["density_keep_mask"])
    scale = summary["scale"]
    normalized = summary["normalized"]

    second = _plot_pca(
        axes[0],
        result["normalized_pca"],
        normalized["pc_variance_ratio"],
        starts,
        "Centered sphere PCA",
        retained,
    )
    figure.colorbar(second, ax=axes[0], label="Window start (s)")
    _plot_diagram(
        axes[1],
        result["normalized_diagram"],
        f"Normalized H1 (PS={normalized.get('periodicity_score', 0):.3f})",
    )
    axes[2].imshow(
        result["normalized_distance"], cmap="magma", origin="lower", vmin=0, vmax=2
    )
    axes[2].set(title="Sphere-normalized distance", xlabel="Window", ylabel="Window")

    figure.suptitle(f"{title}",fontsize=12)
      
    figure.tight_layout()
    path = Path(output) / "topology_diagnostic.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path

