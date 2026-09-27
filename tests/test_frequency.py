import numpy as np

from prism.topology import TopologySettings, estimate_period


def periodic_translation_masks(fps=20.0, duration=8.0, frequency=1.5):
    times = np.arange(int(fps * duration), dtype=float) / fps
    masks = np.zeros((len(times), 48, 64), dtype=bool)
    centers = 31 + np.rint(8 * np.sin(2 * np.pi * frequency * times)).astype(int)
    for index, center in enumerate(centers):
        masks[index, 14:35, center - 5 : center + 6] = True
    return masks, times


def test_lagged_ssm_recovers_known_frequency():
    masks, times = periodic_translation_masks()
    settings = TopologySettings(
        frequency_method="lagged_ssm",
        min_period_sec=0.4,
        max_period_sec=3.0,
        min_cycles=2.5,
    )
    result, _ = estimate_period(masks, times, settings)
    assert result["source"] == "automatic_lagged_ssm"
    assert result["reliable"]
    assert abs(result["frequency_hz"] - 1.5) / 1.5 < 0.05


def test_welch_remains_an_explicit_ablation():
    masks, times = periodic_translation_masks()
    result, _ = estimate_period(
        masks,
        times,
        TopologySettings(frequency_method="welch_pc", max_period_sec=3.0),
    )
    assert result["source"] == "automatic_welch_pc"

