"""
Kaggle for ML Engineers - Production-Grade ML Competition Framework.

This package provides a comprehensive toolkit for machine learning competitions,
including feature engineering, model pipelines, ensembling, and advanced tactics
like pseudo-labeling and model distillation.

Author: Amer Hussein
Version: 1.0.0
License: MIT

Example:
    >>> from kaggle_ml import CompetitionLogger, SafeTargetEncoder
    >>> from kaggle_ml import OOFPipeline, StackingEnsemble
    >>> 
    >>> # Initialize logger
    >>> logger = CompetitionLogger(experiment_name="my_experiment")
    >>> 
    >>> # Create OOF pipeline
    >>> pipeline = OOFPipeline(model=lgb.LGBMClassifier(), cv=5)
    >>> oof_preds = pipeline.fit_predict(X_train, y_train)
"""

__version__ = "1.0.0"
__author__ = "Amer Hussein"
__license__ = "MIT"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Utils
    "CompetitionLogger",
    "set_seed",
    "timer",
    "reduce_memory_usage",
    
    # Features
    "SafeTargetEncoder",
    "create_groupby_features",
    "create_groupby_features_gpu",
    
    # Models
    "OOFPipeline",
    "CrossValidator",
    
    # Ensembling
    "StackingEnsemble",
    "hill_climbing_optimization",
    "blend_predictions",
    
    # Tactics
    "PseudoLabeler",
    "ModelDistiller",
]

# Import core utilities
from src.utils.logger import CompetitionLogger, set_seed, timer, reduce_memory_usage

# Import feature engineering modules
try:
    from src.features.encoding import SafeTargetEncoder
    from src.features.groupby import create_groupby_features, create_groupby_features_gpu
except ImportError as e:
    import warnings
    warnings.warn(f"Feature engineering modules not available: {e}")
    SafeTargetEncoder = None
    create_groupby_features = None
    create_groupby_features_gpu = None

# Import model modules
try:
    from src.models.oof_pipeline import OOFPipeline, CrossValidator
except ImportError as e:
    import warnings
    warnings.warn(f"Model modules not available: {e}")
    OOFPipeline = None
    CrossValidator = None

# Import ensembling modules
try:
    from src.ensembling.stacking import (
        StackingEnsemble,
        hill_climbing_optimization,
        blend_predictions,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Ensembling modules not available: {e}")
    StackingEnsemble = None
    hill_climbing_optimization = None
    blend_predictions = None

# Import tactics modules
try:
    from src.tactics.pseudo_labeling import PseudoLabeler
    from src.tactics.distillation import ModelDistiller
except ImportError as e:
    import warnings
    warnings.warn(f"Tactics modules not available: {e}")
    PseudoLabeler = None
    ModelDistiller = None


def get_version() -> str:
    """Return the current version of the package.
    
    Returns:
        str: The version string in semantic versioning format.
    """
    return __version__


def get_info() -> dict:
    """Return package information.
    
    Returns:
        dict: Dictionary containing package metadata.
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
    }
