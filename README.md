<p align="center">
  <img src="docs/assets/logo.png" width="440"
       alt="PRISM: interpretable measurement of stereotypical movement">
</p>

<p align="center">
  <strong>Interpretable measurement of movement from tracked-person video</strong>
</p>

PRISM converts a continuously tracked person mask and its camera-corrected optical flow into four complementary movement coordinates:

- **Intensity** — typical instantaneous motion magnitude relative to the torso.
- **Extent** — spatial variation of the tracked silhouette.
- **Frequency** — dominant cycles per second.
- **Periodicity** — strength of dominant recurrent organization.

These coordinates are intentionally distinct. A movement can be fast but spatially restricted, large but slow, recurrent at a stable frequency but
inconsistent in cycle shape, or strongly periodic without being especially intense.

The repository has two layers:

1. `prism/` contains the production pipeline used for real SMM video.
2. `prism/validation/` contains controlled Skeleton Lab, multiview, and mask-aligned pose experiments that call the same production measurement functions.

Skeleton Lab is a validation dependency rather than part of the real-video pipeline.

## Installation

Create the production environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[topology,stats,dev]"
```

Install the validation dependencies only when reproducing the controlled experiments:

```bash
python -m pip install -e ".[validation,dev]"
```

DAM4SAM and its checkpoint remain external because tracking is normally run in a separate CUDA environment. The production measurement pipeline begins from the saved full-video tracks.

## Measurements

| Coordinate | Primary estimator | Units |
| --- | --- | --- |
| Intensity | 75th percentile of camera-corrected, torso-relative masked DIS flow, summarized by the temporal median | approximate mask heights/s |
| Extent | framewise RMS radius of the scale-normalized binary-mask orbit | normalized mask-distance units |
| Frequency | interpolated lagged self-similarity of the signed masked-flow trajectory | Hz |
| Periodicity | dominant $H_1$ loop of normalized signed-flow windows, minus a second-loop penalty | unitless |

### Spatial extent

For valid vectorized masks $x_t$ and their mean $\bar{x}=N^{-1}\sum_t x_t$, PRISM defines

$$
E = \left[\frac{1}{N}\sum_{t=1}^{N}\frac{\lVert x_t-\bar{x}\rVert_2^2}{\widetilde A}\right]^{1/2},
$$

where $\widetilde A$ is the median observed mask area. Equivalently, $E$ is the RMS pairwise mask separation.

### Periodicity

For finite $H_1$ classes $(b_i,d_i)$ ordered by decreasing persistence $m_i=d_i-b_i$, PRISM defines

$$
PS = \exp(-b_1/2)\frac{m_1}{\sqrt{3}} - \exp(-b_2/2)\frac{m_2}{\sqrt{3}}.
$$

The first term rewards a dominant persistent loop; the second penalizes a competing loop of comparable persistence.

## Real-video workflow

```bash
# 1. Full-video tracking in the CUDA/DAM4SAM environment
python -m prism.run_dam4sam_batch --help

# 2. Restrict full-video tracks to annotated bouts
python -m prism.prepare_dam4sam_bouts \
  --tracks-root results \
  --episodes-root annotated_runs/episode_runs \
  --output-root prism_bouts

# 3. Camera-corrected optical flow, intensity, and signed-flow states
prism-flow \
  --runs-root prism_bouts \
  --episodes-root annotated_runs/episode_runs \
  --camera-correction ransac_similarity \
  --region-mode flow_consistent \
  --force

# 4. Flow frequency/periodicity and framewise mask-RMS extent
prism-topology \
  --runs-root prism_bouts \
  --d 23 \
  --n-points 300 \
  --min-period-sec 0.40 \
  --max-period-sec 5.0 \
  --min-cycles 2.5 \
  --force

# 5. Assemble the canonical bout-level table
prism-assemble \
  --runs-root prism_bouts \
  --output prism_bouts/analysis.csv
```

The assembled table contains the columns:

```text
prism_intensity
prism_extent
prism_frequency_hz
prism_periodicity
prism_period_sec
prism_torso_translation_speed
```

Failed or ineligible measurements are recorded as unavailable rather than assigned zero.

## Skeleton Lab validation

Run the controlled benchmark from the repository root after installing the `validation` extra:

```bash
python scripts/generate_skeleton_lab_benchmark.py \
  --output-root output/prism_benchmark

python scripts/run_skeleton_lab_benchmark.py \
  --manifest output/prism_benchmark/benchmark_manifest.csv \
  --output output/prism_benchmark/results.csv \
  --camera-correction synthetic

python scripts/summarize_skeleton_lab_benchmark.py \
  --input output/prism_benchmark/results.csv \
  --out output/prism_benchmark/summary
```

The validation pipeline uses the exact synthetic camera transformation rather than estimated RANSAC registration. This isolates the measurement pipeline
from camera-registration error. Viewpoint remains an explicit nuisance because two-dimensional projection can suppress three-dimensional motion.

## Multiview 3D validation

The multiview experiment compares exact Skeleton Lab trajectories with calibrated four-camera reconstruction, an informative clear view, random single views, and an unfavorable view.

```bash
python scripts/export_skeleton_lab_3d.py \
  --bundles-root output/prism_benchmark/bundles \
  --output-root output/prism_multiview/trajectories

python scripts/run_multiview_construct_recovery.py \
  --input output/prism_multiview/trajectories \
  --output output/prism_multiview/construct_recovery_rows.csv \
  --summary-root output/prism_multiview/construct_recovery \
  --random-views 20 \
  --bootstrap-replicates 2000 \
  --seed 20261001

python scripts/run_multiview_regularity_recovery.py \
  --input output/prism_multiview/trajectories \
  --output-root output/prism_multiview/regularity \
  --bootstrap-replicates 2000 \
  --seed 20261001
```

The primary paired outputs are `paired_view_comparison.csv` and `regularity_paired_view_comparison.csv`.

## Mask-aligned pose agreement

Pose is inferred only inside crops derived from the frozen DAM4SAM masks and is evaluated over the same retained frame pairs as optical flow. Pose is treated
as a complementary convergent measurement, not as ground truth.

```bash
python scripts/run_mask_aligned_pose_baseline.py \
  --features prism_bouts/analysis.csv \
  --runs-root prism_bouts \
  --output-root prism_pose_validation \
  --bootstrap-replicates 2000 \
  --seed 20260930
```

The output `pose_agreement_correlations.csv` reports pooled, behavior-specific, and within-participant/behavior Spearman associations with
participant-clustered confidence intervals. The settings export records the exact flow and pose columns used in each comparison.

## Tests

```bash
python -m pytest -q
```

The tests cover body-height normalization, torso-translation removal, signed-flow lagged-self-similarity frequency recovery, framewise RMS mask extent, and the periodicity score.

## Repository layout

```text
PRISM/
├── prism/                         # production package
│   ├── optical_flow.py
│   ├── plotting/
│   ├── topology/
│   └── validation/
│       ├── skeleton_lab/          # controlled 2D benchmark
│       ├── multiview/             # calibrated single-/multiview benchmark
│       └── pose_baseline/         # mask-aligned real-video comparison
├── scripts/                       # reproducible experiment entry points
├── examples/                      # benchmark configurations
├── tests/                         # scientific and integration tests
├── docs/                          # methods and reproducibility notes
└── pyproject.toml
```
