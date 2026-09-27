"""Interpretable measurements of stereotypical motor movement from video."""

from .optical_flow import (
    OpticalFlowSettings,
    analyze_optical_flow_arrays,
    analyze_optical_flow_episode,
)
from .topology import TopologySettings, analyze_masks, estimate_period

__all__ = [
    "OpticalFlowSettings",
    "TopologySettings",
    "analyze_masks",
    "analyze_optical_flow_arrays",
    "analyze_optical_flow_episode",
    "estimate_period",
]

__version__ = "0.2.0"
