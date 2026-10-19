import numpy as np
from scipy.optimize import brentq

from prism.topology.core import cycle_return_distance, persistence_summary
from prism.topology.flow import _period_from_distance, _window_distances
from prism.topology.masks import TopologySettings

T = np.arange(720) / 60
B = W = 1.
H = 3.
NU = .37 * np.sqrt(2)
LEVELS = (0., .15, .3, .6)


def observe(a, l, f, eps, phase, modphase):
    theta = 2 * np.pi * f * T + phase + eps * np.sin(2 * np.pi * NU * T + modphase)
    velocity = a * (2 * np.pi * f + eps * 2 * np.pi * NU * np.cos(2 * np.pi * NU * T + modphase)) * np.cos(theta)
    x = a * np.sin(theta)
    delta = abs(x[:, None] - x[None, :])
    assert delta.max() < W and W * l / (B + W * l) > .25
    gram = l * (W - delta) * np.outer(velocity, velocity) / H ** 2
    norm = np.diag(gram)
    distance = np.sqrt(np.maximum(norm[:, None] + norm[None, :] - 2 * gram, 0) / max(np.median(norm), 1e-12))
    np.fill_diagonal(distance, 0)
    period = _period_from_distance(distance, T, TopologySettings(flow_lag_profile_power=1))
    return dict(intensity=float(np.median(abs(velocity)) / H),
                extent=float(np.sqrt(l * delta.mean() / (B + W * l))),
                frequency=period['frequency_hz'],
                spectral_concentration=period['spectral_concentration']), gram


def matched(seed, eps):
    rng = np.random.default_rng(2026101500 + seed)
    phase, modphase = rng.uniform(0, 2 * np.pi, 2)
    baseline, _ = observe(.1, .8, 1., 0., phase, modphase)

    def at_frequency(f):
        unit, _ = observe(.1, .8, f, eps, phase, modphase)
        a = .1 * baseline['intensity'] / unit['intensity']
        theta = 2 * np.pi * f * T + phase + eps * np.sin(2 * np.pi * NU * T + modphase)
        x = a * np.sin(theta)
        mean_diff = abs(x[:, None] - x[None, :]).mean()
        target = baseline['extent'] ** 2
        l = target * B / (mean_diff - target * W)
        measured, gram = observe(a, l, f, eps, phase, modphase)
        return measured, gram, a, l

    f = 1. if eps == 0 else brentq(lambda f: at_frequency(f)[0]['frequency'] - baseline['frequency'], .9, 1.1, xtol=1e-9)
    measured, gram, a, l = at_frequency(f)
    errors = {key: abs(measured[key] / baseline[key] - 1) for key in ['intensity', 'extent', 'frequency']}
    return dict(seed=seed, epsilon=eps, a=a, l=l, carrier_frequency=f,
                initial_phase=phase, modulation_phase=modphase, **measured,
                **{'match_error_' + key: value for key, value in errors.items()}), gram


def score(row, gram):
    starts, _, distance = _window_distances(gram, T, 1 / row['frequency'], 23, 300)
    summary, _ = persistence_summary(distance, normalized_cloud=True)
    result = dict(row)
    result['ps'] = summary['periodicity_score']
    result['cycle_return_distance'] = cycle_return_distance(distance, starts, 1 / row['frequency'])
    result['windows_retained'] = len(distance)
    return result
