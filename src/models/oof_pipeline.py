"""
Out-of-fold (OOF) prediction pipeline for ML competitions.

This module provides OOFPipeline and CrossValidator classes for generating
out-of-fold predictions and performing cross-validation with proper
train/validation splits.

Author: Amer Hussein
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class OOFPipeline:
    """Out-of-fold prediction pipeline for generating OOF predictions.
    
    This class wraps a model and cross-validation strategy to generate
    out-of-fold predictions, which are essential for stacking ensembles
    and unbiased model evaluation.
    
    Args:
        model: Base model with fit/predict interface (sklearn-compatible).
        cv: Cross-validation splitter or number of folds.
        stratified: Whether to use stratified splitting (for classification).
        random_state: Random seed for reproducibility.
        eval_metric: Metric function for evaluation.
        use_proba: Whether to use predict_proba for classification.
        early_stopping: Whether to use early stopping (for supported models).
        early_stopping_rounds: Number of rounds for early stopping.
    
    Attributes:
        model: The base model.
        oof_predictions: Out-of-fold predictions from fit_predict.
        fold_models: List of fitted models for each fold.
        fold_scores: List of validation scores for each fold.
    
    Example:
        >>> import lightgbm as lgb
        >>> pipeline = OOFPipeline(
        ...     model=lgb.LGBMClassifier(n_estimators=100),
        ...     cv=5,
        ...     stratified=True
        ... )
        >>> oof_preds = pipeline.fit_predict(X_train, y_train)
        >>> test_preds = pipeline.predict(X_test)
        >>> print(f"OOF AUC: {roc_auc_score(y_train, oof_preds):.4f}")
    """
    
    def __init__(
        self,
        model: Any,
        cv: Union[int, Any] = 5,
        stratified: bool = True,
        random_state: int = 42,
        eval_metric: Optional[Callable] = None,
        use_proba: bool = True,
        early_stopping: bool = True,
        early_stopping_rounds: int = 100,
    ) -> None:
        """Initialize the OOFPipeline.
        
        Args:
            model: Base model with sklearn-compatible interface.
            cv: Number of folds or CV splitter.
            stratified: Use stratified splitting for classification.
            random_state: Random seed.
            eval_metric: Custom evaluation metric function.
            use_proba: Use predict_proba for classification.
            early_stopping: Enable early stopping.
            early_stopping_rounds: Early stopping patience.
        """
        self.model = model
        self.cv = cv
        self.stratified = stratified
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.use_proba = use_proba
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        
        self.oof_predictions: Optional[np.ndarray] = None
        self.fold_models: List[Any] = []
        self.fold_scores: List[float] = []
        self._is_fitted = False
        self._n_classes: Optional[int] = None
    
    def _clone_model(self) -> Any:
        """Clone the base model for a new fold.
        
        Returns:
            Cloned model instance.
        """
        try:
            return clone(self.model)
        except Exception:
            # Fallback for models that don't support sklearn clone
            import copy
            return copy.deepcopy(self.model)
    
    def _get_cv_splitter(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        groups: Optional[np.ndarray] = None,
    ) -> Any:
        """Get the cross-validation splitter.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            groups: Group labels for grouped CV.
        
        Returns:
            CV splitter instance.
        """
        if not isinstance(self.cv, int):
            return self.cv
        
        n_splits = self.cv
        
        if groups is not None:
            return GroupKFold(n_splits=n_splits)
        
        # Check if classification problem
        is_classification = len(np.unique(y)) <= 20 or self.stratified
        
        if is_classification and self.stratified:
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            return KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
    
    def _predict_fold(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Make predictions for a single fold.
        
        Args:
            model: Fitted model.
            X: Feature matrix.
        
        Returns:
            Predictions array.
        """
        if self.use_proba and hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)
            # Return probability of positive class for binary
            if preds.shape[1] == 2:
                return preds[:, 1]
            return preds
        else:
            return model.predict(X)
    
    def fit_predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        groups: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Fit models and generate out-of-fold predictions.
        
        Args:
            X: Training features.
            y: Training targets.
            groups: Group labels for grouped CV.
            eval_set: Optional validation set for early stopping.
            verbose: Whether to print progress.
        
        Returns:
            np.ndarray: Out-of-fold predictions.
        """
        X = X.copy() if isinstance(X, pd.DataFrame) else np.array(X)
        y = y.values if isinstance(y, pd.Series) else np.array(y)
        
        cv_splitter = self._get_cv_splitter(X, y, groups)
        
        # Initialize OOF predictions
        if self.use_proba and len(np.unique(y)) > 2:
            self._n_classes = len(np.unique(y))
            self.oof_predictions = np.zeros((len(y), self._n_classes))
        else:
            self.oof_predictions = np.zeros(len(y))
        
        self.fold_models = []
        self.fold_scores = []
        
        fold_idx = 0
        for train_idx, valid_idx in cv_splitter.split(X, y, groups):
            fold_idx += 1
            if verbose:
                print(f"Fold {fold_idx}/{cv_splitter.get_n_splits()}")
            
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            
            # Clone model for this fold
            model = self._clone_model()
            
            # Fit with early stopping if supported
            fit_params = {}
            if self.early_stopping:
                if hasattr(model, "fit") and "early_stopping_rounds" in model.fit.__code__.co_varnames:
                    fit_params["early_stopping_rounds"] = self.early_stopping_rounds
                    fit_params["eval_set"] = [(X_valid, y_valid)]
                    fit_params["verbose"] = False
            
            try:
                model.fit(X_train, y_train, **fit_params)
            except TypeError:
                # Fallback if early stopping params not supported
                model.fit(X_train, y_train)
            
            # Generate OOF predictions for validation set
            valid_preds = self._predict_fold(model, X_valid)
            
            if self.oof_predictions.ndim == 2:
                self.oof_predictions[valid_idx] = valid_preds
            else:
                self.oof_predictions[valid_idx] = valid_preds
            
            # Compute fold score
            if self.eval_metric is not None:
                score = self.eval_metric(y_valid, valid_preds)
            elif len(np.unique(y)) <= 2:
                score = roc_auc_score(y_valid, valid_preds)
            else:
                score = -mean_squared_error(y_valid, valid_preds)
            
            self.fold_scores.append(score)
            self.fold_models.append(model)
            
            if verbose:
                print(f"  Fold {fold_idx} score: {score:.6f}")
        
        self._is_fitted = True
        
        if verbose:
            mean_score = np.mean(self.fold_scores)
            std_score = np.std(self.fold_scores)
            print(f"\nMean CV score: {mean_score:.6f} (+/- {std_score:.6f})")
        
        return self.oof_predictions
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        averaging: str = "mean",
    ) -> np.ndarray:
        """Make predictions on test data by averaging fold predictions.
        
        Args:
            X: Test features.
            averaging: Averaging method ('mean', 'median', 'geometric').
        
        Returns:
            np.ndarray: Averaged predictions.
        
        Raises:
            RuntimeError: If fit_predict has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit_predict before predict")
        
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        
        # Collect predictions from all folds
        fold_predictions = []
        for model in self.fold_models:
            preds = self._predict_fold(model, X)
            fold_predictions.append(preds)
        
        fold_predictions = np.array(fold_predictions)
        
        # Average predictions
        if averaging == "mean":
            return np.mean(fold_predictions, axis=0)
        elif averaging == "median":
            return np.median(fold_predictions, axis=0)
        elif averaging == "geometric":
            from scipy.stats import gmean
            return gmean(fold_predictions, axis=0)
        else:
            raise ValueError(f"Unknown averaging method: {averaging}")
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Get average feature importances across all folds.
        
        Returns:
            np.ndarray: Average feature importances, or None if not available.
        """
        if not self.fold_models:
            return None
        
        importances = []
        for model in self.fold_models:
            if hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)
            elif hasattr(model, "coef_"):
                importances.append(np.abs(model.coef_).flatten())
        
        if not importances:
            return None
        
        return np.mean(importances, axis=0)


class CrossValidator:
    """Cross-validation utility with multiple strategies.
    
    This class provides a unified interface for different cross-validation
    strategies including stratified, grouped, and time-based splitting.
    
    Args:
        strategy: CV strategy ('kfold', 'stratified', 'grouped', 'time').
        n_folds: Number of folds.
        shuffle: Whether to shuffle data before splitting.
        random_state: Random seed.
        stratify_col: Column to stratify by (for stratified CV).
        group_col: Column containing group labels (for grouped CV).
        time_col: Column containing timestamps (for time-based CV).
    
    Example:
        >>> cv = CrossValidator(strategy='stratified', n_folds=5)
        >>> for train_idx, valid_idx in cv.split(X, y):
        ...     X_train, X_valid = X[train_idx], X[valid_idx]
        ...     # Train and evaluate model
    """
    
    def __init__(
        self,
        strategy: str = "stratified",
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        stratify_col: Optional[str] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        """Initialize the CrossValidator.
        
        Args:
            strategy: CV strategy name.
            n_folds: Number of folds.
            shuffle: Whether to shuffle.
            random_state: Random seed.
            stratify_col: Column for stratification.
            group_col: Column for grouping.
            time_col: Column for time-based splitting.
            purge_gap: Gap between train and test for purged CV.
            embargo_pct: Embargo percentage for purged CV.
        """
        self.strategy = strategy
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify_col = stratify_col
        self.group_col = group_col
        self.time_col = time_col
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate train/test indices for cross-validation.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            groups: Group labels.
        
        Yields:
            Tuple of (train_indices, test_indices).
        """
        # Extract stratification array
        stratify = None
        if self.strategy == "stratified" and y is not None:
            stratify = y
            if isinstance(y, pd.Series) and self.stratify_col:
                stratify = y[self.stratify_col] if self.stratify_col in y.index else y
        
        # Extract groups
        if groups is None and self.group_col is not None:
            if isinstance(X, pd.DataFrame) and self.group_col in X.columns:
                groups = X[self.group_col].values
        
        # Create appropriate splitter
        if self.strategy == "kfold":
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            return splitter.split(X)
        
        elif self.strategy == "stratified":
            if stratify is None:
                warnings.warn("Stratified CV requested but no stratification array provided. Using KFold.")
                splitter = KFold(
                    n_splits=self.n_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
            else:
                splitter = StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
            return splitter.split(X, stratify)
        
        elif self.strategy == "grouped":
            if groups is None:
                raise ValueError("Grouped CV requires group labels")
            splitter = GroupKFold(n_splits=self.n_folds)
            return splitter.split(X, y, groups)
        
        elif self.strategy == "time":
            splitter = TimeSeriesSplit(n_splits=self.n_folds)
            return splitter.split(X)
        
        elif self.strategy == "purged":
            return self._purged_split(X, y)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _purged_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        """Generate purged cross-validation splits for time series.
        
        Purged CV removes observations that are too close in time to
        prevent information leakage.
        
        Args:
            X: Feature matrix.
            y: Target vector.
        
        Yields:
            Tuple of (train_indices, test_indices).
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_folds
        
        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            
            # Apply embargo to test set
            embargo_size = int((test_end - test_start) * self.embargo_pct)
            test_start += embargo_size
            
            # Create train set with purge gap
            train_indices = list(range(max(0, test_start - self.purge_gap)))
            train_indices.extend(range(min(n_samples, test_end + self.purge_gap), n_samples))
            
            test_indices = list(range(test_start, test_end))
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self) -> int:
        """Get the number of splits.
        
        Returns:
            int: Number of CV folds.
        """
        return self.n_folds


def get_default_metrics(task: str = "classification") -> Dict[str, Callable]:
    """Get default metrics for a task type.
    
    Args:
        task: Task type ('classification' or 'regression').
    
    Returns:
        Dictionary of metric names and functions.
    """
    if task == "classification":
        return {
            "auc": roc_auc_score,
            "accuracy": accuracy_score,
            "f1": lambda y, p: f1_score(y, (p > 0.5).astype(int)),
        }
    else:
        return {
            "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
            "mae": mean_absolute_error,
            "r2": r2_score,
        }
