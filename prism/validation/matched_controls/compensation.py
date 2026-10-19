from prism.topology.core import persistence_summary
from prism.topology.flow import _window_distances

from .eulerian import T, observe


def evaluate(row):
    measured, gram = observe(row['a'], row['l'], row['carrier_frequency'], 0.,
                             row['initial_phase'], row['modulation_phase'])
    result = dict(row)
    for name, frequency in [('shared_period', row['frequency']), ('own_period', measured['frequency'])]:
        _, _, distance = _window_distances(gram, T, 1 / frequency, 23, 300)
        summary, _ = persistence_summary(distance, normalized_cloud=True)
        result['compensation_ps_' + name] = summary['periodicity_score']
    return result
