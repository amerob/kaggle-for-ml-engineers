"""
Logging utilities for ML competitions with MLflow integration.

This module provides a comprehensive logging system for tracking experiments,
including MLflow integration, seed setting, timing decorators, and memory
optimization utilities.

Author: Amer Hussein
"""

import os
import time
import random
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Install with: pip install mlflow")


class CompetitionLogger:
    """Logger class for ML competition experiments with MLflow integration.
    
    This class provides a comprehensive logging system for tracking machine
    learning experiments, including parameter logging, metric tracking,
    artifact storage, and model versioning.
    
    Args:
        experiment_name: Name of the experiment.
        tracking_uri: URI for MLflow tracking server. If None, uses local storage.
        tags: Optional dictionary of tags to associate with the experiment.
        log_level: Logging level for console output (default: INFO).
    
    Attributes:
        experiment_name: The name of the current experiment.
        run_id: The active MLflow run ID.
        start_time: Timestamp when the logger was initialized.
    
    Example:
        >>> logger = CompetitionLogger(
        ...     experiment_name="kaggle_competition",
        ...     tracking_uri="file:./mlruns"
        ... )
        >>> 
        >>> # Using as context manager
        >>> with logger:
        ...     logger.log_params({"learning_rate": 0.01, "n_estimators": 100})
        ...     logger.log_metrics({"auc": 0.95, "accuracy": 0.92})
        ...     logger.log_artifact("model.pkl")
        ...     logger.log_model(model, "model")
        >>> 
        >>> # Or manual start/end
        >>> logger.start_run()
        >>> logger.log_params({"seed": 42})
        >>> logger.end_run()
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the CompetitionLogger.
        
        Args:
            experiment_name: Name of the experiment.
            tracking_uri: URI for MLflow tracking server.
            tags: Optional dictionary of tags.
            log_level: Logging level for console output.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.tags = tags or {}
        self.run_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self._active = False
        
        # Setup console logger
        self._setup_console_logger(log_level)
        
        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
        else:
            self.logger.warning("MLflow not available. Logging to console only.")
    
    def _setup_console_logger(self, log_level: str) -> None:
        """Setup console logging handler.
        
        Args:
            log_level: Logging level string.
        """
        self.logger = logging.getLogger(f"CompetitionLogger.{self.experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(console_handler)
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                self.logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                self.logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            self.logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def __enter__(self) -> "CompetitionLogger":
        """Context manager entry.
        
        Returns:
            CompetitionLogger: The logger instance.
        """
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit.
        
        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value if an error occurred.
            exc_tb: Exception traceback if an error occurred.
        
        Returns:
            bool: False to propagate exceptions.
        """
        if exc_type is not None:
            self.logger.error(f"Exception occurred: {exc_val}")
            if MLFLOW_AVAILABLE and self._active:
                mlflow.set_tag("error", str(exc_val))
        
        self.end_run()
        return False
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run.
        
        Returns:
            str: The run ID.
        """
        self.start_time = datetime.now()
        
        if MLFLOW_AVAILABLE:
            try:
                run = mlflow.start_run(run_name=run_name)
                self.run_id = run.info.run_id
                self._active = True
                
                # Set default tags
                mlflow.set_tag("start_time", self.start_time.isoformat())
                for key, value in self.tags.items():
                    mlflow.set_tag(key, value)
                
                self.logger.info(f"Started MLflow run: {self.run_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to start MLflow run: {e}")
                raise
        else:
            self.run_id = f"local_{int(time.time())}"
            self._active = True
            self.logger.info(f"Started local run: {self.run_id}")
        
        return self.run_id
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._active:
            self.logger.warning("No active run to end")
            return
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tag("end_time", end_time.isoformat())
                mlflow.set_tag("duration_seconds", duration)
                mlflow.end_run()
                self.logger.info(f"Ended MLflow run: {self.run_id} (duration: {duration:.2f}s)")
            except Exception as e:
                self.logger.error(f"Error ending MLflow run: {e}")
        else:
            self.logger.info(f"Ended local run: {self.run_id} (duration: {duration:.2f}s)")
        
        self._active = False
        self.run_id = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run.
        
        Args:
            params: Dictionary of parameter names and values.
        
        Example:
            >>> logger.log_params({
            ...     "learning_rate": 0.01,
            ...     "n_estimators": 100,
            ...     "max_depth": 6
            ... })
        """
        if not self._active:
            self.logger.warning("No active run. Call start_run() first.")
            return
        
        # Convert non-serializable values to strings
        params = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v 
                  for k, v in params.items()}
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_params(params)
            except Exception as e:
                self.logger.error(f"Error logging params: {e}")
        
        # Always log to console
        for key, value in params.items():
            self.logger.info(f"Param: {key} = {value}")
    
    def log_metrics(
        self, 
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for the metrics.
        
        Example:
            >>> logger.log_metrics({
            ...     "train_auc": 0.95,
            ...     "valid_auc": 0.92,
            ...     "train_loss": 0.15
            ... }, step=100)
        """
        if not self._active:
            self.logger.warning("No active run. Call start_run() first.")
            return
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                self.logger.error(f"Error logging metrics: {e}")
        
        # Always log to console
        prefix = f"[Step {step}] " if step is not None else ""
        for key, value in metrics.items():
            self.logger.info(f"{prefix}Metric: {key} = {value:.6f}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact (file) to the current run.
        
        Args:
            local_path: Path to the local file.
            artifact_path: Optional directory within the artifact store.
        
        Example:
            >>> logger.log_artifact("model.pkl")
            >>> logger.log_artifact("confusion_matrix.png", "plots")
        """
        if not self._active:
            self.logger.warning("No active run. Call start_run() first.")
            return
        
        if not os.path.exists(local_path):
            self.logger.error(f"Artifact not found: {local_path}")
            return
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(local_path, artifact_path)
                self.logger.info(f"Logged artifact: {local_path}")
            except Exception as e:
                self.logger.error(f"Error logging artifact: {e}")
        else:
            self.logger.info(f"Would log artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log all artifacts in a directory.
        
        Args:
            local_dir: Path to the local directory.
            artifact_path: Optional directory within the artifact store.
        """
        if not self._active:
            self.logger.warning("No active run. Call start_run() first.")
            return
        
        if not os.path.isdir(local_dir):
            self.logger.error(f"Directory not found: {local_dir}")
            return
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifacts(local_dir, artifact_path)
                self.logger.info(f"Logged artifacts from: {local_dir}")
            except Exception as e:
                self.logger.error(f"Error logging artifacts: {e}")
        else:
            self.logger.info(f"Would log artifacts from: {local_dir}")
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set tags for the current run.
        
        Args:
            tags: Dictionary of tag names and values.
        
        Example:
            >>> logger.set_tags({
            ...     "model_type": "lightgbm",
            ...     "version": "v1.0",
            ...     "team": "alpha"
            ... })
        """
        if not self._active:
            self.logger.warning("No active run. Call start_run() first.")
            return
        
        # Convert values to strings
        tags = {k: str(v) for k, v in tags.items()}
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tags(tags)
            except Exception as e:
                self.logger.error(f"Error setting tags: {e}")
        
        # Always log to console
        for key, value in tags.items():
            self.logger.info(f"Tag: {key} = {value}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log a model to the current run.
        
        Args:
            model: The model object to log.
            artifact_path: Path within the artifact store.
            registered_model_name: Optional name for model registry.
        
        Example:
            >>> logger.log_model(model, "model", "my_model_v1")
        """
        if not self._active:
            self.logger.warning("No active run. Call start_run() first.")
            return
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                )
                self.logger.info(f"Logged model to: {artifact_path}")
            except Exception as e:
                self.logger.error(f"Error logging model: {e}")
                # Try generic logging
                try:
                    mlflow.pyfunc.log_model(artifact_path, python_model=model)
                except Exception as e2:
                    self.logger.error(f"Fallback model logging also failed: {e2}")
        else:
            self.logger.info(f"Would log model to: {artifact_path}")
    
    def log_dict(self, dictionary: Dict[str, Any], filename: str = "config.json") -> None:
        """Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log.
            filename: Name of the output file.
        """
        import json
        
        if not self._active:
            self.logger.warning("No active run. Call start_run() first.")
            return
        
        temp_path = f"/tmp/{filename}"
        try:
            with open(temp_path, "w") as f:
                json.dump(dictionary, f, indent=2, default=str)
            self.log_artifact(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    This function sets the random seed for Python's random module, NumPy,
    and common ML libraries like PyTorch and TensorFlow if available.
    
    Args:
        seed: The random seed value (default: 42).
    
    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logging.info(f"Random seed set to: {seed}")


@contextmanager
def timer(name: str = "Operation", logger: Optional[logging.Logger] = None) -> None:
    """Context manager for timing code blocks.
    
    Args:
        name: Name of the operation being timed.
        logger: Optional logger to use. If None, prints to stdout.
    
    Example:
        >>> with timer("Data loading"):
        ...     df = pd.read_csv("data.csv")
        >>> # Output: [Data loading] done in 1.23s
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        msg = f"[{name}] done in {elapsed:.3f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def reduce_memory_usage(
    df: pd.DataFrame,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Reduce memory usage of a pandas DataFrame by downcasting types.
    
    This function automatically converts numeric columns to the smallest
    possible dtype that can hold the data without loss of precision.
    
    Args:
        df: Input DataFrame.
        verbose: Whether to print memory usage information.
        logger: Optional logger for output.
    
    Returns:
        pd.DataFrame: DataFrame with optimized memory usage.
    
    Example:
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        >>> df_optimized = reduce_memory_usage(df)
        >>> # Memory usage reduced from X MB to Y MB
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
    
    msg = f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)"
    
    if logger:
        logger.info(msg)
    elif verbose:
        print(msg)
    
    return df


def log_system_info(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Log system information for reproducibility.
    
    Args:
        logger: Optional logger for output.
    
    Returns:
        Dict containing system information.
    """
    import platform
    import sys
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    
    # Add optional dependencies
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    
    try:
        import mlflow
        info["mlflow_version"] = mlflow.__version__
    except ImportError:
        pass
    
    msg = "System Info:\n" + "\n".join(f"  {k}: {v}" for k, v in info.items())
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return info
