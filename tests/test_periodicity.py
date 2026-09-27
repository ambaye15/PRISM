import numpy as np

from prism.topology import periodicity_score_from_diagram


def test_periodicity_score_uses_birth_weights_and_second_loop_penalty():
    diagram = np.asarray([[0.2, 1.4], [0.4, 1.1], [0.1, 0.2]])
    expected = (
        np.exp(-0.2 / 2.0) * 1.2 / np.sqrt(3.0)
        - np.exp(-0.4 / 2.0) * 0.7 / np.sqrt(3.0)
    )
    assert np.isclose(periodicity_score_from_diagram(diagram), expected)


def test_periodicity_score_has_no_penalty_when_only_one_loop_exists():
    diagram = np.asarray([[0.2, 1.4]])
    expected = np.exp(-0.2 / 2.0) * 1.2 / np.sqrt(3.0)
    assert np.isclose(periodicity_score_from_diagram(diagram), expected)

