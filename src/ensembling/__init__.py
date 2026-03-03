"""
Ensembling modules for the Kaggle ML framework.

This module provides ensembling techniques including stacking, blending,
and hill climbing optimization for combining multiple model predictions.
"""

from src.ensembling.stacking import (
    StackingEnsemble,
    hill_climbing_optimization,
    blend_predictions,
)

__all__ = [
    "StackingEnsemble",
    "hill_climbing_optimization",
    "blend_predictions",
]
