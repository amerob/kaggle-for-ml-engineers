"""
Advanced tactics modules for the Kaggle ML framework.

This module provides advanced competition tactics including pseudo-labeling
and model distillation for improving model performance.
"""

from src.tactics.pseudo_labeling import PseudoLabeler
from src.tactics.distillation import ModelDistiller

__all__ = [
    "PseudoLabeler",
    "ModelDistiller",
]
