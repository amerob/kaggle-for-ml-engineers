"""
Feature engineering modules for the Kaggle ML framework.

This module provides tools for feature engineering including target encoding,
groupby aggregations, and other competition-proven techniques.
"""

from src.features.encoding import SafeTargetEncoder
from src.features.groupby import create_groupby_features, create_groupby_features_gpu

__all__ = [
    "SafeTargetEncoder",
    "create_groupby_features",
    "create_groupby_features_gpu",
]
