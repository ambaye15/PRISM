# Reproduction

Run commands from the repository root, in the order shown within each workflow. Continue only after the preceding stage succeeds. Synthetic inputs are generated from source; participant data and saved paper results are not bundled.

## Setup

Use Python 3.11 or 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[paper,dev]'
```

The `paper` extra includes the synthetic, exemplar, plotting, and statistical dependencies. It installs Skeleton Lab at the commit pinned in `pyproject.toml`, requiring Git and access to that repository. For pose inference, also install:

```bash
python -m pip install -e '.[pose]'
```

DAM4SAM and its checkpoint belong in the tracker's own compatible environment. Keep this source checkout: the workflows below also use its `scripts/`, `tools/`, and `examples/` directories.

### Settings to check

| Setting | Workflow |
| --- | --- |
| Lag-distance power | All commands below use power 2. |
| Minimum estimated cycles | Real-video main analysis uses 0 in both topology extraction and analysis; 2.5 is a separate sensitivity analysis. |
| Window sampling | All use `d=23`, and 300 windows. |
| Extent support | Real-video commands explicitly require at least two valid masks.|

Use a new results directory after changing a protocol. Keep manifests, settings, dependency versions, and trained checkpoints with the outputs. Fresh training and different environments can change numerical results.

## 1. Synthetic construct recovery

Generate the full benchmark:

```bash
python scripts/generate_skeleton_lab_benchmark.py \
  --output-root output/prism_benchmark --specs-root examples/specs
```

Expect **1,890 conditions**: 1,125 clean and 765 perturbed. Then measure and summarize:

```bash
python scripts/run_skeleton_lab_benchmark.py \
  --manifest output/prism_benchmark/benchmark_manifest.csv \
  --output output/prism_benchmark/results.csv \
  --max-dimension 256 --dimension 23 --n-points 300 \
  --camera-correction synthetic

python scripts/summarize_skeleton_lab_benchmark.py \
  --input output/prism_benchmark/results.csv \
  --out output/prism_benchmark/summary
```

The summary directory contains tables and figures. Synthetic camera correction uses the known transformation, isolating downstream measurements from registration error.

## 2. Matched and rendered repetition controls

Run the exact-field and compensation experiments together:

```bash
python -m prism.validation.matched_controls run \
  --results-dir runs/matched_controls --workers 4
```

This produces **32 matched conditions and 24 compensation conditions**, with `summary.json`, `RESULTS.md`, tables, and figures. Matching intensity, extent, and frequency precedes persistence computation. The experiment needs no archived results.

Run the separate rendered controls:

```bash
python scripts/run_oscillator_regularity.py \
  --output-root output/oscillator_regularity --specs-root examples/specs \
  --behaviors body_rocking --levels 0 .25 .5 .75 --seeds 0 1 2 \
  --duration-sec 12 --warmup-sec 2 --fps 20 --d 23 --n-points 300 --videos
```

Inspect `manifest.csv`, `results.csv`, and `regularity_summary.csv`. These controls pass through rendering and DIS and use estimated and reference periods. The design has 24 records but 13 distinct numeric bundles; these are not 24 independent movements.

## 3. Real-video profiling

Supply recordings, annotations, person initialization boxes, episode clips, and metadata. Track each full recording continuously, then extract its annotated bouts.

In the DAM4SAM environment:

```bash
python -m prism.run_dam4sam_batch \
  --manifest inputs/tracking_manifest.json --dam4sam-root /path/to/DAM4SAM \
  --output-root results --boxes-file inputs/dam4sam_boxes.json
```

The manifest contains a `videos` list with unique `video_id`, video `path`, and per-frame `annotations`. Labels are 0=none, 1=rock, 2=flap, and 3=flap--rock. Relative paths resolve from the manifest's parent directory's parent unless `--project-root` is supplied. Use `configs/dam4sam_boxes.example.json` for the initialization-box format and replace the example coordinates.

Each `results/<video_id>/` must contain `dam4sam_track_summary.json`, `dam4sam_video_masks_packed.npy`, and `dam4sam_video_mask_valid.npy`. Episode inputs need the adapter's metadata and an `assets/*_raw.mp4` clip per episode. Metadata supplies `analysis_start_frame`, `analysis_end_frame_exclusive`, `core_start_frame`, and `core_end_frame`; preserve the source-video correspondence.

In the PRISM environment:

```bash
python -m prism.prepare_dam4sam_bouts \
  --tracks-root results --episodes-root annotated_runs/episode_runs \
  --output-root prism_bouts

prism-flow --runs-root prism_bouts --episodes-root annotated_runs/episode_runs \
  --backend dis --camera-correction ransac_similarity --region-mode flow_consistent

prism-topology --runs-root prism_bouts --d 23 --n-points 300 \
  --min-period-sec 0.40 --max-period-sec 5.0 \
  --min-cycles 0 --minimum-extent-frames 2

prism-assemble --runs-root prism_bouts --output prism_bouts/analysis.csv
```

If needed, add `--metadata inputs/episode_metadata.csv` to assembly. It must have unique matching `episode_id` values and authoritative participant, behavior, session, and timing fields. Supplied metadata replaces overlapping metadata columns. Add `--force` to flow or topology when replacing existing measurements after changing settings.

Analyze the main profiles and the cycle-count sensitivity separately:

```bash
python scripts/analyze_p001_p004_and_matched_pairs.py \
  --input prism_bouts/analysis.csv --output-root output/real_video/main \
  --participants P001 P004 --min-cycles 0

python scripts/analyze_p001_p004_and_matched_pairs.py \
  --input prism_bouts/analysis.csv --output-root output/real_video/sensitivity_25 \
  --participants P001 P004 --min-cycles 2.5
```

Outputs include `analysis_ready.csv`, availability and frequency-QC tables, session profiles, matched pairs, and figures. Frequency eligibility uses the saved reliability flag, mask coverage, cycle rule, and a ceiling at 80% of Nyquist. A permissive downstream rule cannot restore estimates rejected during extraction; rerun topology with `--min-cycles 0 --force` if necessary.

## 4. Pose agreement and audits

Use the assembled features and prepared bout tracks. Supply the selected pose weights locally or allow their acquisition by the inference library.

```bash
python scripts/run_mask_aligned_pose_baseline.py \
  --features prism_bouts/analysis.csv --runs-root prism_bouts \
  --output-root prism_pose_validation --model yolo11m-pose.pt \
  --bootstrap-replicates 2000 --seed 20260930
```

Add `--videos-root /path/to/recordings` if stored video paths have moved. Pose is inferred in padded mask crops and compared with flow on exactly shared valid frame pairs. Main outputs are `prism_pose_merged.csv`, `pose_agreement_correlations.csv`, `pose_availability.csv`, and `pose_baseline_settings.json`.

Audit whichever sources are available:

```bash
python scripts/audit_pose_quality.py \
  --sam-features prism_pose_validation/prism_pose_merged.csv \
  --sam-tracks-root prism_pose_validation/pose_tracks \
  --asdmotion-pickle inputs/asdmotion.pkl --output-root output/pose_audits
```

Omit unavailable sources. For AQSM, add `--aqsm-pickle inputs/aqsm.pkl` and `--aqsm-image-aspect` followed by the verified image width/height ratio. At least one source is required.

The audit writes coverage, diagnostic candidates, processing sensitivity, and settings. AQSM and SAM-crop pose cover the same cohort; ASDMotion is external. Confidence scores differ across systems, and flagged discontinuities are not verified errors.

## 5. Profile-conditioned movement

Run the nested training-library study used for the larger-model illustration:

```bash
python -m prism.exemplars.sample_size run \
  --results-dir runs/exemplar_sample_size \
  --training-sizes 144 576 1440 --workers 4 --illustrate

python -m prism.exemplars.sample_size validate --results-dir runs/exemplar_sample_size
```

The study shares **54 test targets** across three models and remeasures 162 neural outputs. All sizes use the 144-library IQR scales. Results include `summary.csv`, `summary.json`, `sample_size.pdf`, and per-size model/evaluation directories. The main illustration is `train_1440/profile_compact.pdf` and `train_1440/profile_to_motion.mp4`.

For a standalone run, use:

```bash
python -m prism.exemplars library --results-dir runs/new_exemplars --lag-profile-power 1 --workers 4
python -m prism.exemplars train --results-dir runs/new_exemplars
python -m prism.exemplars decode --results-dir runs/new_exemplars --workers 4
python -m prism.exemplars validate --results-dir runs/new_exemplars
python -m prism.exemplars illustrate --results-dir runs/new_exemplars
python tools/summarize_exemplar_run.py --results-dir runs/new_exemplars --output-root output/exemplar_tables
```

This produces 144 library profiles, 54 test profiles, and 54 newly rendered neural outputs. Inspect failed records before validation or illustration. Use `decode --force` after resolving a rendering or measurement failure.

Use the sample-size validator for expanded libraries. Independently rerunning ordinary training or decoding on a study leaf resets its scoring to that leaf's IQR; rerun the study workflow before comparing sizes. Keep the trained checkpoint for exact model reuse.

## 6. Supplementary multiview experiments

Start from the bundles generated in step 1:

```bash
python scripts/export_skeleton_lab_3d.py \
  --bundles-root output/prism_benchmark/bundles \
  --output-root input_data/prism_multiview/trajectories

python tools/prepare_multiview_inputs.py \
  --input-root input_data/prism_multiview --output-root output/multiview_inputs
```

The full export should contain **1,380 trajectories with zero skipped bundles**. Preparation recovers the known geometry labels needed for calibration. Trajectory file count does not imply independent motion count.

```bash
python scripts/run_multiview_construct_recovery.py \
  --input output/multiview_inputs/trajectories --output output/multiview/raw_rows.csv \
  --summary-root output/multiview/construct_recovery \
  --random-views 20 --keypoint-noise-px 1 --missing-probability .05 \
  --bootstrap-replicates 2000 --seed 20261001

python scripts/run_multiview_regularity_recovery.py \
  --input output/multiview_inputs/trajectories --output-root output/multiview/regularity \
  --levels 0 .1 .2 .4 --keypoint-noise-px 1 --missing-probability .05 \
  --bootstrap-replicates 2000 --seed 20261001
```

The paired outputs are `paired_view_comparison.csv` and `regularity_paired_view_comparison.csv`. These are landmark-based comparisons, separate from the production mask/flow measurements. Favorable-view selection uses the generating geometry. No noninferiority script is needed for the reported analyses.

## Checks and saved results

```bash
python -m pytest -q
```