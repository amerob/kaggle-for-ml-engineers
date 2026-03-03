"""
Model distillation for knowledge transfer in ML competitions.

This module provides ModelDistiller class for distilling knowledge from
an ensemble of teacher models into a smaller student model.

Author: Amer Hussein
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.special import softmax


class ModelDistiller:
    """Model distillation for knowledge transfer.
    
    Model distillation transfers knowledge from a complex ensemble of teacher
    models to a simpler student model. The student learns from the soft
    probabilities produced by teachers, which contain more information than
    hard labels.
    
    Args:
        teacher_models: List of teacher models or dictionary mapping names to models.
        student_model: Student model to train.
        temperature: Temperature for softening probability distributions.
                      Higher values produce softer distributions.
        alpha: Weight for distillation loss (1-alpha for ground truth loss).
        ensemble_method: Method for combining teacher predictions ('mean', 'weighted').
        teacher_weights: Optional weights for teacher ensemble.
    
    Attributes:
        fitted_student: The fitted student model.
        teacher_predictions: Predictions from teacher models.
    
    Example:
        >>> import lightgbm as lgb
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> 
        >>> teachers = {
        ...     'lgb': lgb.LGBMClassifier(n_estimators=500),
        ...     'rf': RandomForestClassifier(n_estimators=200),
        ... }
        >>> student = lgb.LGBMClassifier(n_estimators=100)
        >>> 
        >>> distiller = ModelDistiller(
        ...     teacher_models=teachers,
        ...     student_model=student,
        ...     temperature=3.0,
        ...     alpha=0.7
        ... )
        >>> 
        >>> # Distill knowledge
        >>> distilled_model = distiller.distill(
        ...     X_train, y_train, X_test
        ... )
        >>> 
        >>> # Make predictions with student model
        >>> predictions = distilled_model.predict(X_test)
    """
    
    def __init__(
        self,
        teacher_models: Union[List[Any], Dict[str, Any]],
        student_model: Any,
        temperature: float = 3.0,
        alpha: float = 0.7,
        ensemble_method: str = "mean",
        teacher_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the ModelDistiller.
        
        Args:
            teacher_models: Teacher models (list or dict).
            student_model: Student model.
            temperature: Temperature for softening.
            alpha: Weight for distillation loss.
            ensemble_method: Ensemble method for teachers.
            teacher_weights: Optional weights for teachers.
        """
        if isinstance(teacher_models, list):
            self.teacher_models = {f"teacher_{i}": m for i, m in enumerate(teacher_models)}
        else:
            self.teacher_models = teacher_models
        
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.ensemble_method = ensemble_method
        self.teacher_weights = teacher_weights
        
        self.fitted_student: Optional[Any] = None
        self.fitted_teachers: Dict[str, Any] = {}
        self.teacher_predictions: Optional[np.ndarray] = None
        self._is_fitted = False
    
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
    
    def _temperature_scale(
        self,
        logits: np.ndarray,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Apply temperature scaling to logits or probabilities.
        
        Temperature scaling softens the probability distribution:
        - T > 1: softer distribution (more uncertainty)
        - T = 1: original distribution
        - T < 1: harder distribution (more confident)
        
        Args:
            logits: Logits or probabilities.
            temperature: Temperature value. Uses instance temperature if None.
        
        Returns:
            Temperature-scaled probabilities.
        """
        if temperature is None:
            temperature = self.temperature
        
        # Check if input is already probabilities
        if np.all((logits >= 0) & (logits <= 1)):
            # Convert to logits
            logits = np.log(logits + 1e-8)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert back to probabilities
        if scaled_logits.ndim == 1:
            # Binary classification
            probs = 1 / (1 + np.exp(-scaled_logits))
            return np.vstack([1 - probs, probs]).T
        else:
            # Multi-class classification
            return softmax(scaled_logits, axis=1)
    
    def _get_predictions(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        use_proba: bool = True,
    ) -> np.ndarray:
        """Get predictions from a model.
        
        Args:
            model: Fitted model.
            X: Feature matrix.
            use_proba: Whether to use predict_proba.
        
        Returns:
            Predictions array.
        """
        if use_proba and hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return model.predict(X)
    
    def fit_teachers(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_valid: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_valid: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: bool = False,
    ) -> "ModelDistiller":
        """Fit all teacher models.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_valid: Optional validation features.
            y_valid: Optional validation labels.
            verbose: Whether to print progress.
        
        Returns:
            ModelDistiller: Self with fitted teachers.
        """
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_train = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        
        if X_valid is not None:
            X_valid = X_valid.values if isinstance(X_valid, pd.DataFrame) else np.array(X_valid)
        if y_valid is not None:
            y_valid = y_valid.values if isinstance(y_valid, pd.Series) else np.array(y_valid)
        
        self.fitted_teachers = {}
        
        for name, model_template in self.teacher_models.items():
            if verbose:
                print(f"Training teacher: {name}")
            
            model = self._clone_model(model_template)
            
            # Fit with validation set if provided and supported
            if X_valid is not None and y_valid is not None:
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False
                    )
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            self.fitted_teachers[name] = model
        
        return self
    
    def generate_teacher_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        use_temperature: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """Generate ensemble predictions from teacher models.
        
        Args:
            X: Feature matrix.
            use_temperature: Whether to apply temperature scaling.
            verbose: Whether to print progress.
        
        Returns:
            Ensemble predictions from teachers.
        
        Raises:
            RuntimeError: If teachers have not been fitted.
        """
        if not self.fitted_teachers:
            raise RuntimeError("Teachers must be fitted first. Call fit_teachers().")
        
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        
        # Collect predictions from all teachers
        teacher_preds = []
        
        for name, model in self.fitted_teachers.items():
            preds = self._get_predictions(model, X, use_proba=True)
            
            if use_temperature:
                preds = self._temperature_scale(preds)
            
            teacher_preds.append(preds)
        
        # Ensemble predictions
        if self.ensemble_method == "mean":
            ensemble_preds = np.mean(teacher_preds, axis=0)
        elif self.ensemble_method == "weighted":
            if self.teacher_weights is None:
                # Equal weights
                weights = [1.0 / len(teacher_preds)] * len(teacher_preds)
            else:
                weights = [
                    self.teacher_weights.get(name, 1.0)
                    for name in self.fitted_teachers.keys()
                ]
                total = sum(weights)
                weights = [w / total for w in weights]
            
            ensemble_preds = np.average(teacher_preds, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        self.teacher_predictions = ensemble_preds
        
        return ensemble_preds
    
    def distill(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_student: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_student: Optional[Union[pd.Series, np.ndarray]] = None,
        use_soft_labels: bool = True,
        verbose: bool = False,
    ) -> Any:
        """Distill knowledge from teachers to student.
        
        Args:
            X_train: Training features for teachers.
            y_train: Training labels for teachers.
            X_student: Optional separate data for student training.
                       If None, uses X_train.
            y_student: Optional labels for student data.
                       If None, uses teacher predictions.
            use_soft_labels: Whether to use soft labels from teachers.
            verbose: Whether to print progress.
        
        Returns:
            Fitted student model.
        """
        # Fit teachers if not already fitted
        if not self.fitted_teachers:
            self.fit_teachers(X_train, y_train, verbose=verbose)
        
        # Determine student training data
        if X_student is None:
            X_student = X_train
        if y_student is None:
            # Generate teacher predictions for student training
            y_student = self.generate_teacher_predictions(X_student, verbose=verbose)
        
        X_student = X_student.values if isinstance(X_student, pd.DataFrame) else np.array(X_student)
        
        if isinstance(y_student, (pd.Series, pd.DataFrame)):
            y_student = y_student.values
        
        # Convert soft labels to hard labels if needed
        if use_soft_labels and y_student.ndim == 2:
            # Use soft labels directly
            student_labels = y_student
        elif y_student.ndim == 2:
            # Convert to hard labels
            student_labels = np.argmax(y_student, axis=1)
        else:
            student_labels = y_student
        
        if verbose:
            print("Training student model...")
        
        # Train student model
        self.fitted_student = self._clone_model(self.student_model)
        
        # Handle different label types
        if student_labels.ndim == 2 and hasattr(self.fitted_student, "predict_proba"):
            # For models that support sample weights, we can use soft labels
            # Otherwise, use hard labels
            try:
                self.fitted_student.fit(X_student, np.argmax(student_labels, axis=1))
            except Exception as e:
                if verbose:
                    print(f"Failed to fit with soft labels: {e}")
                self.fitted_student.fit(X_student, student_labels)
        else:
            self.fitted_student.fit(X_student, student_labels)
        
        self._is_fitted = True
        
        if verbose:
            print("Distillation complete!")
        
        return self.fitted_student
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        use_student: bool = True,
    ) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            use_student: Whether to use student model (True) or teachers (False).
        
        Returns:
            Predictions array.
        """
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        
        if use_student:
            if self.fitted_student is None:
                raise RuntimeError("Student model has not been fitted")
            
            if hasattr(self.fitted_student, "predict_proba"):
                return self.fitted_student.predict_proba(X)
            return self.fitted_student.predict(X)
        else:
            return self.generate_teacher_predictions(X)
    
    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        metric: Callable,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Evaluate teachers and student.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            metric: Evaluation metric function.
            verbose: Whether to print results.
        
        Returns:
            Dictionary of evaluation scores.
        """
        y_test = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)
        
        results = {}
        
        # Evaluate each teacher
        for name, model in self.fitted_teachers.items():
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_test)
                if preds.shape[1] == 2:
                    preds = preds[:, 1]
                else:
                    preds = np.argmax(preds, axis=1)
            else:
                preds = model.predict(X_test)
            
            score = metric(y_test, preds)
            results[f"teacher_{name}"] = score
            
            if verbose:
                print(f"Teacher {name}: {score:.6f}")
        
        # Evaluate student
        if self.fitted_student is not None:
            if hasattr(self.fitted_student, "predict_proba"):
                preds = self.fitted_student.predict_proba(X_test)
                if preds.shape[1] == 2:
                    preds = preds[:, 1]
                else:
                    preds = np.argmax(preds, axis=1)
            else:
                preds = self.fitted_student.predict(X_test)
            
            score = metric(y_test, preds)
            results["student"] = score
            
            if verbose:
                print(f"Student: {score:.6f}")
        
        return results


def create_distillation_dataset(
    teacher_models: List[Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_unlabeled: Union[pd.DataFrame, np.ndarray],
    temperature: float = 3.0,
    sample_ratio: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dataset for distillation using unlabeled data.
    
    This function generates soft labels from teacher models on unlabeled data
    and combines it with the original labeled data.
    
    Args:
        teacher_models: List of teacher models.
        X_train: Labeled training features.
        y_train: Labeled training targets.
        X_unlabeled: Unlabeled features.
        temperature: Temperature for softening.
        sample_ratio: Ratio of unlabeled data to use.
        random_state: Random seed.
    
    Returns:
        Tuple of (combined features, combined soft labels).
    """
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
    y_train = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    X_unlabeled = X_unlabeled.values if isinstance(X_unlabeled, pd.DataFrame) else np.array(X_unlabeled)
    
    # Fit teachers
    fitted_teachers = []
    for model in teacher_models:
        fitted_model = clone(model) if hasattr(model, "get_params") else model
        fitted_model.fit(X_train, y_train)
        fitted_teachers.append(fitted_model)
    
    # Sample unlabeled data
    if sample_ratio < 1.0:
        n_sample = int(len(X_unlabeled) * sample_ratio)
        np.random.seed(random_state)
        sample_idx = np.random.choice(len(X_unlabeled), size=n_sample, replace=False)
        X_unlabeled = X_unlabeled[sample_idx]
    
    # Generate teacher predictions
    teacher_preds = []
    for model in fitted_teachers:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_unlabeled)
        else:
            preds = model.predict(X_unlabeled)
            # Convert to one-hot if needed
            if preds.ndim == 1:
                unique_labels = np.unique(y_train)
                one_hot = np.zeros((len(preds), len(unique_labels)))
                for i, pred in enumerate(preds):
                    idx = np.where(unique_labels == pred)[0][0]
                    one_hot[i, idx] = 1
                preds = one_hot
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = np.log(preds + 1e-8)
            scaled_logits = logits / temperature
            preds = softmax(scaled_logits, axis=1)
        
        teacher_preds.append(preds)
    
    # Average predictions
    soft_labels = np.mean(teacher_preds, axis=0)
    
    # Combine labeled and unlabeled data
    X_combined = np.vstack([X_train, X_unlabeled])
    
    # Create soft labels for training data
    n_classes = soft_labels.shape[1]
    y_train_soft = np.zeros((len(y_train), n_classes))
    for i, label in enumerate(y_train):
        y_train_soft[i, int(label)] = 1.0
    
    y_combined = np.vstack([y_train_soft, soft_labels])
    
    return X_combined, y_combined
