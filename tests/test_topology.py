import numpy as np
from scipy.spatial.distance import pdist, squareform

from prism.topology import orbit_radius_from_distances


def test_orbit_radius_equals_rms_distance_to_centroid():
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    distance = squareform(pdist(points))
    centered = points - points.mean(axis=0, keepdims=True)
    expected = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    assert np.isclose(orbit_radius_from_distances(distance), expected)

