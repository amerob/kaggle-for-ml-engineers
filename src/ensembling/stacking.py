"""
Stacking ensemble and optimization utilities for ML competitions.

This module provides StackingEnsemble class for stacking multiple models,
hill climbing optimization for ensemble weights, and blending utilities.

Author: Amer Hussein
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from scipy.optimize import minimize
from scipy.stats import gmean


class StackingEnsemble:
    """Stacking ensemble for combining multiple base models.
    
    Stacking (stacked generalization) uses out-of-fold predictions from
    base models as features for a meta-model, which learns to optimally
    combine the base model predictions.
    
    Args:
        base_models: Dictionary of model name to model instance.
        meta_model: Meta-learner model. If None, uses LogisticRegression
                   for classification or Ridge for regression.
        n_folds: Number of folds for generating OOF predictions.
        stratified: Whether to use stratified splitting.
        random_state: Random seed.
        use_proba: Whether to use predict_proba for classification.
        passthrough: Whether to include original features in meta-features.
    
    Attributes:
        fitted_base_models: Dictionary of fitted base models for each fold.
        fitted_meta_model: Fitted meta-learner.
        oof_meta_features: Out-of-fold meta-features used for training.
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import lightgbm as lgb
        >>> import xgboost as xgb
        >>> 
        >>> base_models = {
        ...     'lgb': lgb.LGBMClassifier(),
        ...     'xgb': xgb.XGBClassifier(),
        ...     'rf': RandomForestClassifier()
        ... }
        >>> 
        >>> ensemble = StackingEnsemble(
        ...     base_models=base_models,
        ...     meta_model=LogisticRegression(),
        ...     n_folds=5
        ... )
        >>> 
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Optional[Any] = None,
        n_folds: int = 5,
        stratified: bool = True,
        random_state: int = 42,
        use_proba: bool = True,
        passthrough: bool = False,
    ) -> None:
        """Initialize the StackingEnsemble.
        
        Args:
            base_models: Dictionary of model name to model instance.
            meta_model: Meta-learner model.
            n_folds: Number of CV folds.
            stratified: Use stratified splitting.
            random_state: Random seed.
            use_proba: Use predict_proba for classification.
            passthrough: Include original features.
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_state = random_state
        self.use_proba = use_proba
        self.passthrough = passthrough
        
        self.fitted_base_models: Dict[str, List[Any]] = {name: [] for name in base_models}
        self.fitted_meta_model: Optional[Any] = None
        self.oof_meta_features: Optional[np.ndarray] = None
        self._is_fitted = False
        self._is_classification: Optional[bool] = None
        self._n_classes: Optional[int] = None
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model.
        
        Args:
            model: Model to clone.
        
        Returns:
            Cloned model.
        """
        try:
            return clone(model)
        except Exception:
            import copy
            return copy.deepcopy(model)
    
    def _get_predictions(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Get predictions from a model.
        
        Args:
            model: Fitted model.
            X: Feature matrix.
        
        Returns:
            Predictions array.
        """
        if self.use_proba and hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)
            if preds.shape[1] == 2:
                return preds[:, 1]
            return preds
        return model.predict(X)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        groups: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "StackingEnsemble":
        """Fit the stacking ensemble.
        
        Args:
            X: Training features.
            y: Training targets.
            groups: Group labels for grouped CV.
            verbose: Whether to print progress.
        
        Returns:
            StackingEnsemble: Fitted ensemble.
        """
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Determine task type
        self._is_classification = len(np.unique(y)) <= 20
        self._n_classes = len(np.unique(y)) if self._is_classification else None
        
        # Setup CV
        if self.stratified and self._is_classification:
            cv = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            cv = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
        
        # Initialize meta-features
        n_models = len(self.base_models)
        if self._is_classification and self._n_classes and self._n_classes > 2 and self.use_proba:
            meta_feature_dim = n_models * self._n_classes
        else:
            meta_feature_dim = n_models
        
        if self.passthrough:
            meta_feature_dim += X.shape[1]
        
        self.oof_meta_features = np.zeros((len(X), meta_feature_dim))
        
        # Generate OOF predictions for each base model
        model_idx = 0
        for name, model_template in self.base_models.items():
            if verbose:
                print(f"Training base model: {name}")
            
            oof_preds = np.zeros(len(X))
            if self._is_classification and self._n_classes and self._n_classes > 2 and self.use_proba:
                oof_preds = np.zeros((len(X), self._n_classes))
            
            fold_idx = 0
            for train_idx, valid_idx in cv.split(X, y, groups):
                fold_idx += 1
                
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]
                
                # Clone and fit model
                model = self._clone_model(model_template)
                model.fit(X_train, y_train)
                
                # Store fitted model
                self.fitted_base_models[name].append(model)
                
                # Generate OOF predictions
                preds = self._get_predictions(model, X_valid)
                
                if oof_preds.ndim == 2:
                    oof_preds[valid_idx] = preds
                else:
                    oof_preds[valid_idx] = preds
            
            # Store OOF predictions in meta-features
            if oof_preds.ndim == 2:
                self.oof_meta_features[:, model_idx:model_idx + self._n_classes] = oof_preds
                model_idx += self._n_classes
            else:
                self.oof_meta_features[:, model_idx] = oof_preds
                model_idx += 1
        
        # Add original features if passthrough
        if self.passthrough:
            self.oof_meta_features[:, -X.shape[1]:] = X
        
        # Fit meta-model
        if self.meta_model is None:
            if self._is_classification:
                self.meta_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            else:
                self.meta_model = Ridge(random_state=self.random_state)
        
        self.fitted_meta_model = self._clone_model(self.meta_model)
        self.fitted_meta_model.fit(self.oof_meta_features, y)
        
        self._is_fitted = True
        
        if verbose:
            if self._is_classification:
                meta_score = roc_auc_score(y, self.fitted_meta_model.predict_proba(self.oof_meta_features)[:, 1])
            else:
                meta_score = -mean_squared_error(y, self.fitted_meta_model.predict(self.oof_meta_features))
            print(f"Meta-model score: {meta_score:.6f}")
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        averaging: str = "mean",
    ) -> np.ndarray:
        """Make predictions on test data.
        
        Args:
            X: Test features.
            averaging: Averaging method for base model predictions.
        
        Returns:
            np.ndarray: Predictions.
        
        Raises:
            RuntimeError: If ensemble has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before predict")
        
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        
        # Generate meta-features for test data
        n_models = len(self.base_models)
        if self._is_classification and self._n_classes and self._n_classes > 2 and self.use_proba:
            meta_feature_dim = n_models * self._n_classes
        else:
            meta_feature_dim = n_models
        
        if self.passthrough:
            meta_feature_dim += X.shape[1]
        
        test_meta_features = np.zeros((len(X), meta_feature_dim))
        
        # Average predictions across folds for each base model
        model_idx = 0
        for name in self.base_models:
            fold_preds = []
            for model in self.fitted_base_models[name]:
                preds = self._get_predictions(model, X)
                fold_preds.append(preds)
            
            fold_preds = np.array(fold_preds)
            
            if averaging == "mean":
                avg_preds = np.mean(fold_preds, axis=0)
            elif averaging == "median":
                avg_preds = np.median(fold_preds, axis=0)
            elif averaging == "geometric":
                avg_preds = gmean(fold_preds, axis=0)
            else:
                raise ValueError(f"Unknown averaging: {averaging}")
            
            if avg_preds.ndim == 2:
                test_meta_features[:, model_idx:model_idx + self._n_classes] = avg_preds
                model_idx += self._n_classes
            else:
                test_meta_features[:, model_idx] = avg_preds
                model_idx += 1
        
        # Add original features if passthrough
        if self.passthrough:
            test_meta_features[:, -X.shape[1]:] = X
        
        # Make predictions with meta-model
        if self._is_classification and self.use_proba and hasattr(self.fitted_meta_model, "predict_proba"):
            preds = self.fitted_meta_model.predict_proba(test_meta_features)
            if preds.shape[1] == 2:
                return preds[:, 1]
            return preds
        
        return self.fitted_meta_model.predict(test_meta_features)


def hill_climbing_optimization(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    metric: Callable = roc_auc_score,
    maximize: bool = True,
    max_iterations: int = 1000,
    step_size: float = 0.01,
    patience: int = 50,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Optimize ensemble weights using hill climbing.
    
    This function finds optimal weights for combining model predictions
    using a greedy hill climbing algorithm.
    
    Args:
        predictions: Dictionary mapping model names to prediction arrays.
        y_true: True target values.
        metric: Metric function to optimize.
        maximize: Whether to maximize (True) or minimize (False) the metric.
        max_iterations: Maximum number of iterations.
        step_size: Step size for weight adjustments.
        patience: Number of iterations without improvement before stopping.
        random_state: Random seed.
        verbose: Whether to print progress.
    
    Returns:
        Tuple of (weights dictionary, best score).
    
    Example:
        >>> predictions = {
        ...     'lgb': lgb_preds,
        ...     'xgb': xgb_preds,
        ...     'cat': cat_preds
        ... }
        >>> weights, score = hill_climbing_optimization(
        ...     predictions, y_valid, roc_auc_score, maximize=True
        ... )
        >>> print(f"Best weights: {weights}, Score: {score:.6f}")
    """
    np.random.seed(random_state)
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    # Initialize equal weights
    weights = {name: 1.0 / n_models for name in model_names}
    
    def get_weighted_prediction(w: Dict[str, float]) -> np.ndarray:
        """Get weighted ensemble prediction."""
        pred = np.zeros(len(y_true))
        for name in model_names:
            pred += w[name] * predictions[name]
        return pred
    
    def evaluate(w: Dict[str, float]) -> float:
        """Evaluate metric with given weights."""
        pred = get_weighted_prediction(w)
        score = metric(y_true, pred)
        return score if maximize else -score
    
    best_score = evaluate(weights)
    best_weights = weights.copy()
    
    no_improvement_count = 0
    
    for iteration in range(max_iterations):
        # Randomly select a model to adjust
        model_to_adjust = np.random.choice(model_names)
        
        # Try increasing and decreasing weight
        direction = np.random.choice([-1, 1])
        
        new_weights = weights.copy()
        new_weights[model_to_adjust] += direction * step_size
        
        # Ensure weights are non-negative and sum to 1
        for name in model_names:
            new_weights[name] = max(0, new_weights[name])
        
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        else:
            continue
        
        # Evaluate new weights
        new_score = evaluate(new_weights)
        
        if new_score > best_score:
            best_score = new_score
            best_weights = new_weights.copy()
            weights = new_weights
            no_improvement_count = 0
            
            if verbose:
                print(f"Iteration {iteration}: New best score = {best_score:.6f}")
        else:
            no_improvement_count += 1
        
        # Early stopping
        if no_improvement_count >= patience:
            if verbose:
                print(f"Early stopping at iteration {iteration}")
            break
    
    # Return with correct sign
    final_score = best_score if maximize else -best_score
    
    return best_weights, final_score


def blend_predictions(
    predictions: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted",
) -> np.ndarray:
    """Blend predictions from multiple models.
    
    Args:
        predictions: Dictionary mapping model names to prediction arrays.
        weights: Optional weights for each model. If None, uses equal weights.
        method: Blending method ('weighted', 'mean', 'median', 'geometric').
    
    Returns:
        np.ndarray: Blended predictions.
    
    Example:
        >>> predictions = {
        ...     'lgb': lgb_preds,
        ...     'xgb': xgb_preds
        ... }
        >>> blended = blend_predictions(predictions, method='mean')
    """
    model_names = list(predictions.keys())
    preds_array = np.array([predictions[name] for name in model_names])
    
    if method == "mean":
        return np.mean(preds_array, axis=0)
    
    elif method == "median":
        return np.median(preds_array, axis=0)
    
    elif method == "geometric":
        return gmean(preds_array, axis=0)
    
    elif method == "weighted":
        if weights is None:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        
        # Normalize weights
        total = sum(weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}
        
        # Compute weighted average
        result = np.zeros(len(preds_array[0]))
        for name in model_names:
            result += normalized_weights[name] * predictions[name]
        
        return result
    
    else:
        raise ValueError(f"Unknown blending method: {method}")


def optimize_weights_scipy(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    metric: Callable = roc_auc_score,
    maximize: bool = True,
    method: str = "SLSQP",
) -> Tuple[np.ndarray, float]:
    """Optimize ensemble weights using scipy optimization.
    
    Args:
        predictions: Dictionary of model predictions.
        y_true: True target values.
        metric: Metric function.
        maximize: Whether to maximize.
        method: Scipy optimization method.
    
    Returns:
        Tuple of (weights array, best score).
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    preds_array = np.array([predictions[name] for name in model_names])
    
    def objective(weights: np.ndarray) -> float:
        """Objective function to minimize."""
        ensemble_pred = np.average(preds_array, axis=0, weights=weights)
        score = metric(y_true, ensemble_pred)
        return -score if maximize else score
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models
    
    result = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
    )
    
    best_weights = result.x
    best_score = -result.fun if maximize else result.fun
    
    return best_weights, best_score
