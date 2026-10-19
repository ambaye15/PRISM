# Measurements

PRISM describes an identified movement bout using four measurements. Duration and inter-bout interval provide temporal context.

| Measurement | Question | Source | Units |
| --- | --- | --- | --- |
| Intensity | How strongly does the person move relative to a torso reference? | Camera-corrected, torso-relative masked flow | Approximate mask heights/s |
| Extent | How much does the silhouette vary over the bout? | Binary person masks in a common crop | Area-normalized mask distance |
| Frequency | How quickly does movement repeat? | Camera-corrected total signed flow | Hz |
| Periodicity score (PS) | How clearly does motion express a dominant repeating cycle? | Normalized windows of total signed flow | Unitless |

Intensity uses torso-relative flow. Frequency and PS use total flow after camera correction. Extent is computed directly from masks and does not depend on the estimated period.

## Intensity

For a valid frame pair, let $v_t(p)$ be camera-corrected flow displacement, $c_t$ the estimated translation of a central mask region, $h_t$ the smoothed mask-height scale, and $\Delta t$ the elapsed time. Over retained pixels $R_t$,

$$
I_t=\frac{1}{h_t\Delta t}quantile_{0.75}\{\lVert v_t(p)-c_t\rVert_2:p\in R_t\},
\qquad I=median_t I_t.
$$

The central region supplies a geometric torso reference; it is not an anatomical pose estimate. Pixel retention checks mask consistency and flow quality. Mask height supplies an image-based scale, so intensity measures projected motion rather than calibrated three-dimensional speed. Total motion and reference-region translation are also saved.

## Extent

Let $x_t$ be a vectorized binary mask, $\bar{x}=N^{-1}\sum_t x_t$, and $\widetilde A$ the median retained mask area. Extent is the RMS variation around the mean silhouette:

$$
E=\left[\frac{1}{N\widetilde A}\sum_{t=1}^{N}\lVert x_t-\bar{x}\rVert_2^2\right]^{1/2}.
$$

Equivalently,

$$
E^2=\frac{1}{2N^2\widetilde A}\sum_{t=1}^{N}\sum_{s=1}^{N}\lVert x_t-x_s\rVert_2^2.
$$

Extent describes silhouette variation, not a joint's maximum reach. Viewpoint and segmentation errors can change it.

## Frequency

Horizontal and vertical flow retain their signs and are pooled on a body grid. PRISM applies confidence weighting and robust clipping, then analyzes an uninterrupted valid run. For the resulting flow-state distances $D_{ij}$, the lag profile is

$$
g_q(\ell)=mean_i D_{i,i+\ell}^{\,q}.
$$

The interpolated spectral peak of this profile supplies frequency $f$ and period $T=1/f$. Power 1 averages distances; power 2 averages squared distances.

The supplied batch runners use `TopologySettings.flow_lag_profile_power=2`. Record this setting with each run; frequency and PS from different definitions should not be treated as interchangeable.

Short records may contain too little repetition for reliable estimation even below Nyquist. The main real-video workflow uses no minimum-cycle cutoff. Duration, estimated cycles, mask coverage, and spectral quality remain relevant.

## PS: regularity of repetition

Frequency gives the repetition rate. PS describes how strongly the analyzed motion supports one dominant repeating cycle. A high score is consistent with clear, regular repetition in this representation.

PRISM forms sliding windows spanning approximately one selected period, centers and normalizes them, and computes their one-dimensional persistent homology. With `d=23`, each window has 24 samples spanning $23T/24$; `n_points` controls the number of windows.

For finite $H_1$ classes $(b_i,d_i)$ ordered by decreasing lifetime $m_i=d_i-b_i$,

$$
PS=e^{-b_1/2}\frac{m_1}{\sqrt{3}}-e^{-b_2/2}\frac{m_2}{\sqrt{3}}.
$$

The first term rewards a persistent loop. The second reduces the score when a competing loop has comparable support. A missing class contributes zero. The implementation uses Vietoris–Rips persistence over coefficient field 47.

PS follows the evolution of the motion through its cycle. It is not simply a check that two states match one period apart, or a direct measure of cycle-duration variability. Timing disruption, changes in cycle shape, competing rhythms, noise, and period selection can all affect it. Window normalization removes overall scale, leaving intensity and extent to describe magnitude.

Keep the returned score, including negative values. PS is not a probability or guaranteed to lie in $[0,1]$. A computed zero differs from an unavailable measurement and does not, by itself, establish that the original movement was aperiodic.

## Timing, outputs, and interpretation

Duration is the bout's elapsed time. The preceding inter-bout interval is current onset minus the previous annotated bout's offset, regardless of behavior. First bouts and negative intervals are unavailable; no interval should be inferred across a recording pause.

| Assembled column | Measurement |
| --- | --- |
| `prism_intensity` | Torso-relative intensity |
| `prism_extent` | Mask RMS extent |
| `prism_frequency_hz` | Numerical frequency estimate |
| `prism_periodicity` | PS |
| `prism_period_sec` | Selected period |
| `prism_torso_translation_speed` | Reference-region translation |
| `duration_sec` | Bout duration |
| `prism_interbout_interval_sec` | Preceding interval |

Missing values remain unavailable. A finite frequency can still fail the analysis eligibility rule; retain its quality fields.

Two bouts can have similar intensity and frequency while differing in extent or PS. The measurements describe recorded movement and have no established conversion to clinical severity. They remain two-dimensional observations, affected by visibility, masks, and flow quality.

A profile also does not specify a unique body, viewpoint, or movement. Retrieval and the neural decoder supply compatible examples. Remeasurement checks whether a generated example reproduces the requested profile.
