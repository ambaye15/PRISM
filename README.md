# PRISM

PRISM measures movement intensity, spatial extent, frequency, periodic organization, and duration from tracked-person video masks.

The repository has two separate layers:

1. `prism/` is the production analysis used on real SMM videos.
2. `prism/validation/skeleton_lab/` runs controlled synthetic validation with
   Skeleton Lab, but calls the same production measurement functions.

Skeleton Lab is a validation dependency.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[topology,stats,dev]"
```

Install the controlled validation environment only when reproducing the synthetic experiments:

```bash
python -m pip install -e ".[validation,dev]"
```

DAM4SAM and its checkpoint remain external because they require their own CUDA environment. See [`docs/real_video_pipeline.md`](docs/real_video_pipeline.md).

## Measurements

| Coordinate | Primary estimator | Units |
| --- | --- | --- |
| Intensity | camera-corrected, torso-centered masked DIS flow p75 | body heights/s |
| Extent | RMS radius of the unnormalized sliding-window point cloud | mask-distance units |
| Frequency | interpolated lagged self-similarity spectrum | Hz |
| Periodicity | birth-weighted dominant H1 loop minus the second-loop penalty | unitless |
| Duration | annotated onset-to-offset interval | s |

Welch frequency is retained as a benchmark comparator, not the primary reported frequency estimator.

For finite H1 classes ordered by decreasing persistence `m_i = d_i - b_i`, periodicity is

$$
PS = exp(-b1 / 2)\frac{m1}{sqrt(3)} - exp(-b2 / 2)\frac{m2}{sqrt(3)}
$$


## Real-video workflow

```bash
# 1. Full-video tracking (CUDA/DAM4SAM environment)
python -m prism.run_dam4sam_batch --help

# 2. Restrict the full-video tracks to annotated bouts
python -m prism.prepare_dam4sam_bouts --help

# 3. Torso-centered masked optical-flow intensity
prism-flow --help

# 4. Lagged-SSM frequency, orbit-radius extent, and SW1PerS periodicity
prism-topology --help

# 5. Canonical bout-level table
prism-assemble --help
```

## Skeleton Lab validation

Run this from the PRISM repository root after installing the `validation` extra. The scripts import Skeleton Lab as a dependency and save all generated data below `output/`.

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

## Tests

```bash
python -m pytest -q
```

The tests check the scientific contract: body-height normalization, torso translation removal, lagged-SSM frequency recovery, and the definition of the orbit radius and periodicity score.


## Repository layout

```text
PRISM/
├── prism/                         # production package
│   ├── optical_flow.py
│   ├── plotting/                 
│   ├── topology/
│   └── validation/skeleton_lab/   # synthetic adapters only
├── scripts/                       # reproducible benchmark entry points
├── examples/                      # benchmark configurations
├── tests/                         # scientific and integration tests
├── docs/                          # methods and reproducibility notes
└── pyproject.toml
```
