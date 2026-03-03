"""
Pseudo-labeling for semi-supervised learning in ML competitions.

This module provides PseudoLabeler class for generating pseudo-labels
from unlabeled data and iteratively improving model performance.

Author: Amer Hussein
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone


class PseudoLabeler:
    """Pseudo-labeling for semi-supervised learning.
    
    Pseudo-labeling is a technique where a model trained on labeled data
    is used to predict labels for unlabeled data. High-confidence predictions
    are then added to the training set, and the model is retrained.
    
    Args:
        model: Base model with fit/predict interface.
        confidence_threshold: Minimum confidence for pseudo-labels (0-1).
        max_iterations: Maximum number of pseudo-labeling iterations.
        sample_ratio: Ratio of pseudo-labeled samples to add each iteration.
        stratified: Whether to use stratified sampling.
        random_state: Random seed.
    
    Attributes:
        iteration_history: List of metrics for each iteration.
        pseudo_labels_: Generated pseudo-labels from last iteration.
    
    Example:
        >>> import lightgbm as lgb
        >>> pseudo_labeler = PseudoLabeler(
        ...     model=lgb.LGBMClassifier(),
        ...     confidence_threshold=0.9,
        ...     max_iterations=3
        ... )
        >>> 
        >>> # Generate pseudo-labels
        >>> X_combined, y_combined = pseudo_labeler.generate_pseudo_labels(
        ...     X_train, y_train, X_test, cv=5
        ... )
        >>> 
        >>> # Or use iterative pseudo-labeling
        >>> final_model = pseudo_labeler.iterative_pseudo_labeling(
        ...     X_train, y_train, X_test
        ... )
    """
    
    def __init__(
        self,
        model: Any,
        confidence_threshold: float = 0.9,
        max_iterations: int = 3,
        sample_ratio: float = 0.5,
        stratified: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialize the PseudoLabeler.
        
        Args:
            model: Base model.
            confidence_threshold: Minimum confidence threshold.
            max_iterations: Maximum iterations.
            sample_ratio: Sample ratio per iteration.
            stratified: Use stratified sampling.
            random_state: Random seed.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.sample_ratio = sample_ratio
        self.stratified = stratified
        self.random_state = random_state
        
        self.iteration_history: List[Dict] = []
        self.pseudo_labels_: Optional[np.ndarray] = None
        self.pseudo_confidences_: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def _clone_model(self) -> Any:
        """Clone the base model.
        
        Returns:
            Cloned model.
        """
        try:
            return clone(self.model)
        except Exception:
            import copy
            return copy.deepcopy(self.model)
    
    def _get_confidences(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Get prediction confidences from model.
        
        Args:
            model: Fitted model.
            X: Feature matrix.
        
        Returns:
            Confidence scores.
        """
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Return max probability as confidence
            return np.max(proba, axis=1)
        else:
            # For models without predict_proba, use prediction magnitude
            preds = model.predict(X)
            # Normalize to [0, 1] if not already
            if np.min(preds) < 0 or np.max(preds) > 1:
                preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-8)
            return preds
    
    def generate_pseudo_labels(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        cv: int = 5,
        eval_metric: Optional[Callable] = None,
        verbose: bool = False,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        """Generate pseudo-labels using cross-validation.
        
        This method uses out-of-fold predictions to generate pseudo-labels
        for unlabeled data, which helps prevent overfitting.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Unlabeled test features.
            cv: Number of CV folds.
            eval_metric: Optional evaluation metric.
            verbose: Whether to print progress.
        
        Returns:
            Tuple of (combined features, combined labels).
        """
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_train = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
        
        # Setup CV
        is_classification = len(np.unique(y_train)) <= 20
        if self.stratified and is_classification:
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            split_iter = kf.split(X_train, y_train)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            split_iter = kf.split(X_train)
        
        # Generate OOF predictions on test set
        test_preds = np.zeros(len(X_test))
        if is_classification and hasattr(self.model, "predict_proba"):
            n_classes = len(np.unique(y_train))
            if n_classes > 2:
                test_preds = np.zeros((len(X_test), n_classes))
        
        test_confidences = np.zeros(len(X_test))
        
        for fold_idx, (train_idx, valid_idx) in enumerate(split_iter):
            if verbose:
                print(f"Fold {fold_idx + 1}/{cv}")
            
            X_tr, X_val = X_train[train_idx], X_train[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]
            
            # Train model
            model = self._clone_model()
            model.fit(X_tr, y_tr)
            
            # Predict on test set
            if hasattr(model, "predict_proba") and test_preds.ndim == 2:
                fold_preds = model.predict_proba(X_test)
                test_preds += fold_preds / cv
                test_confidences += np.max(fold_preds, axis=1) / cv
            else:
                fold_preds = model.predict(X_test)
                test_preds += fold_preds / cv
                test_confidences += np.ones(len(X_test)) / cv
        
        # Store confidences
        self.pseudo_confidences_ = test_confidences
        
        # Determine pseudo-labels based on predictions
        if test_preds.ndim == 2:
            pseudo_labels = np.argmax(test_preds, axis=1)
        else:
            # For regression or binary classification
            if is_classification:
                pseudo_labels = (test_preds > 0.5).astype(int)
            else:
                pseudo_labels = test_preds
        
        self.pseudo_labels_ = pseudo_labels
        
        # Filter by confidence threshold
        confident_mask = test_confidences >= self.confidence_threshold
        n_confident = np.sum(confident_mask)
        
        if verbose:
            print(f"Generated {n_confident} pseudo-labels above threshold {self.confidence_threshold}")
        
        if n_confident == 0:
            warnings.warn("No pseudo-labels met confidence threshold. Returning original data.")
            return X_train, y_train
        
        # Sample pseudo-labeled data
        if self.sample_ratio < 1.0:
            n_sample = int(n_confident * self.sample_ratio)
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(
                np.where(confident_mask)[0],
                size=n_sample,
                replace=False
            )
        else:
            sample_idx = np.where(confident_mask)[0]
        
        # Combine training and pseudo-labeled data
        X_combined = np.vstack([X_train, X_test[sample_idx]])
        y_combined = np.concatenate([y_train, pseudo_labels[sample_idx]])
        
        self._is_fitted = True
        
        if verbose:
            print(f"Combined dataset size: {len(y_combined)} (original: {len(y_train)}, pseudo: {len(sample_idx)})")
        
        return X_combined, y_combined
    
    def iterative_pseudo_labeling(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        X_valid: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_valid: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_metric: Optional[Callable] = None,
        verbose: bool = False,
    ) -> Any:
        """Perform iterative pseudo-labeling.
        
        This method iteratively adds pseudo-labels and retrains the model,
        potentially improving performance with each iteration.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Unlabeled test features.
            X_valid: Optional validation features.
            y_valid: Optional validation labels.
            eval_metric: Optional evaluation metric.
            verbose: Whether to print progress.
        
        Returns:
            Fitted model after iterative pseudo-labeling.
        """
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_train = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
        
        if X_valid is not None:
            X_valid = X_valid.values if isinstance(X_valid, pd.DataFrame) else np.array(X_valid)
        if y_valid is not None:
            y_valid = y_valid.values if isinstance(y_valid, pd.Series) else np.array(y_valid)
        
        current_X, current_y = X_train.copy(), y_train.copy()
        best_model = None
        best_score = float('-inf')
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Train model on current data
            model = self._clone_model()
            model.fit(current_X, current_y)
            
            # Evaluate on validation set if provided
            if X_valid is not None and y_valid is not None and eval_metric is not None:
                if hasattr(model, "predict_proba"):
                    val_preds = model.predict_proba(X_valid)[:, 1]
                else:
                    val_preds = model.predict(X_valid)
                
                score = eval_metric(y_valid, val_preds)
                
                if verbose:
                    print(f"Validation score: {score:.6f}")
                
                if score > best_score:
                    best_score = score
                    best_model = self._clone_model()
                    best_model.fit(current_X, current_y)
            
            # Generate predictions on test set
            if hasattr(model, "predict_proba"):
                test_proba = model.predict_proba(X_test)
                test_confidences = np.max(test_proba, axis=1)
                test_preds = np.argmax(test_proba, axis=1)
            else:
                test_preds = model.predict(X_test)
                test_confidences = np.ones(len(X_test))
            
            # Filter by confidence
            confident_mask = test_confidences >= self.confidence_threshold
            n_confident = np.sum(confident_mask)
            
            if verbose:
                print(f"Pseudo-labels above threshold: {n_confident}")
            
            if n_confident == 0:
                if verbose:
                    print("No more confident pseudo-labels. Stopping.")
                break
            
            # Sample pseudo-labels
            if self.sample_ratio < 1.0:
                n_sample = int(n_confident * self.sample_ratio)
                np.random.seed(self.random_state + iteration)
                sample_idx = np.random.choice(
                    np.where(confident_mask)[0],
                    size=min(n_sample, n_confident),
                    replace=False
                )
            else:
                sample_idx = np.where(confident_mask)[0]
            
            # Add pseudo-labeled data
            current_X = np.vstack([current_X, X_test[sample_idx]])
            current_y = np.concatenate([current_y, test_preds[sample_idx]])
            
            # Store iteration info
            self.iteration_history.append({
                "iteration": iteration + 1,
                "n_pseudo_labels": len(sample_idx),
                "total_size": len(current_y),
            })
        
        # Return best model or last model
        if best_model is not None:
            if verbose:
                print(f"\nBest validation score: {best_score:.6f}")
            return best_model
        
        # Fit final model on all data
        final_model = self._clone_model()
        final_model.fit(current_X, current_y)
        
        self._is_fitted = True
        
        return final_model
    
    def get_pseudo_label_stats(self) -> Dict:
        """Get statistics about pseudo-labeling process.
        
        Returns:
            Dictionary with pseudo-labeling statistics.
        """
        if not self._is_fitted:
            return {"error": "PseudoLabeler has not been fitted yet"}
        
        stats = {
            "n_iterations": len(self.iteration_history),
            "iteration_history": self.iteration_history,
        }
        
        if self.pseudo_confidences_ is not None:
            stats["confidence_stats"] = {
                "mean": float(np.mean(self.pseudo_confidences_)),
                "std": float(np.std(self.pseudo_confidences_)),
                "min": float(np.min(self.pseudo_confidences_)),
                "max": float(np.max(self.pseudo_confidences_)),
            }
        
        return stats


def soft_pseudo_labeling(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    temperature: float = 1.0,
    sample_ratio: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Soft pseudo-labeling using prediction probabilities.
    
    Instead of hard labels, this function uses soft probabilities as labels,
    which can provide more information during training.
    
    Args:
        model: Base model.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        temperature: Temperature for softening predictions.
        sample_ratio: Ratio of samples to include.
        random_state: Random seed.
    
    Returns:
        Tuple of (combined features, soft labels).
    """
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
    y_train = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Get soft predictions
    if hasattr(model, "predict_proba"):
        soft_labels = model.predict_proba(X_test)
        
        # Apply temperature scaling
        if temperature != 1.0:
            soft_labels = np.exp(np.log(soft_labels + 1e-8) / temperature)
            soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
    else:
        # For regression, use predictions directly
        soft_labels = model.predict(X_test).reshape(-1, 1)
    
    # Sample if needed
    if sample_ratio < 1.0:
        n_sample = int(len(X_test) * sample_ratio)
        np.random.seed(random_state)
        sample_idx = np.random.choice(len(X_test), size=n_sample, replace=False)
        X_test = X_test[sample_idx]
        soft_labels = soft_labels[sample_idx]
    
    # Combine
    X_combined = np.vstack([X_train, X_test])
    
    # For soft labels, we need to expand y_train to match dimensions
    if soft_labels.ndim == 2 and soft_labels.shape[1] > 1:
        # Multi-class: one-hot encode y_train
        n_classes = soft_labels.shape[1]
        y_train_soft = np.zeros((len(y_train), n_classes))
        for i, label in enumerate(y_train):
            y_train_soft[i, int(label)] = 1
        y_combined = np.vstack([y_train_soft, soft_labels])
    else:
        y_combined = np.concatenate([y_train, soft_labels.flatten()])
    
    return X_combined, y_combined
