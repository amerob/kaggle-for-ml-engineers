"""
Target encoding with out-of-fold protection for categorical variables.

This module provides SafeTargetEncoder, a robust implementation of target encoding
that prevents data leakage by using out-of-fold encoding during training.

Author: Amer Hussein
"""

import warnings
from typing import List, Optional, Union, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


class SafeTargetEncoder:
    """Safe target encoder with out-of-fold protection.
    
    This encoder implements target encoding with smoothing and optional noise
    to prevent overfitting. It uses out-of-fold encoding during training to
    prevent data leakage, making it safe for cross-validation.
    
    The encoding formula is:
        encoded = (count * mean + smoothing * global_mean) / (count + smoothing)
    
    Args:
        smoothing: Smoothing parameter. Higher values increase regularization.
        min_samples_leaf: Minimum samples required for a category to use its own mean.
        noise: Standard deviation of Gaussian noise to add (0 for no noise).
        handle_unknown: How to handle unknown categories ('value' or 'error').
        handle_missing: How to handle missing values ('value' or 'error').
    
    Attributes:
        encodings_: Dictionary mapping column names to encoding dictionaries.
        global_means_: Dictionary of global means for each target-encoded column.
        fitted_columns_: List of columns that were fitted.
    
    Example:
        >>> encoder = SafeTargetEncoder(smoothing=10.0, noise=0.01)
        >>> 
        >>> # Fit and transform with OOF protection
        >>> X_encoded = encoder.fit_transform(
        ...     df, 
        ...     cols=['category_1', 'category_2'],
        ...     target=df['target'],
        ...     n_folds=5
        ... )
        >>> 
        >>> # Transform test data
        >>> X_test_encoded = encoder.transform(test_df, cols=['category_1', 'category_2'])
    """
    
    def __init__(
        self,
        smoothing: float = 10.0,
        min_samples_leaf: int = 2,
        noise: float = 0.0,
        handle_unknown: str = "value",
        handle_missing: str = "value",
    ) -> None:
        """Initialize the SafeTargetEncoder.
        
        Args:
            smoothing: Smoothing parameter for regularization.
            min_samples_leaf: Minimum samples for category-specific encoding.
            noise: Standard deviation of noise to add (0 for no noise).
            handle_unknown: Strategy for unknown categories.
            handle_missing: Strategy for missing values.
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise = noise
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        
        self.encodings_: Dict[str, Dict] = {}
        self.global_means_: Dict[str, float] = {}
        self.fitted_columns_: List[str] = []
        self._is_fitted = False
    
    def _compute_encoding(
        self, 
        series: pd.Series, 
        target: pd.Series
    ) -> Dict:
        """Compute target encoding for a single column.
        
        Args:
            series: Categorical series to encode.
            target: Target values.
        
        Returns:
            Dictionary mapping category values to encoded values.
        """
        # Compute global mean
        global_mean = target.mean()
        
        # Compute category statistics
        agg = pd.DataFrame({"target": target, "category": series})
        stats = agg.groupby("category")["target"].agg(["count", "mean"])
        
        # Apply smoothing
        smoothed_means = (
            stats["count"] * stats["mean"] + self.smoothing * global_mean
        ) / (stats["count"] + self.smoothing)
        
        # Create encoding dictionary
        encoding = smoothed_means.to_dict()
        
        # Store global mean for unknown categories
        encoding["__global_mean__"] = global_mean
        
        return encoding
    
    def fit(
        self,
        df: pd.DataFrame,
        cols: List[str],
        target: Union[pd.Series, np.ndarray],
    ) -> "SafeTargetEncoder":
        """Fit the encoder on training data.
        
        This method computes the target encodings for each categorical column
        using all training data. Use fit_transform for OOF-protected encoding.
        
        Args:
            df: Training DataFrame.
            cols: List of categorical columns to encode.
            target: Target values.
        
        Returns:
            SafeTargetEncoder: Fitted encoder instance.
        """
        target = pd.Series(target)
        
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            # Compute encoding
            encoding = self._compute_encoding(df[col], target)
            self.encodings_[col] = encoding
            self.global_means_[col] = encoding["__global_mean__"]
        
        self.fitted_columns_ = cols
        self._is_fitted = True
        
        return self
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        cols: List[str],
        target: Union[pd.Series, np.ndarray],
        n_folds: int = 5,
        stratified: bool = True,
        random_state: int = 42,
        return_full_df: bool = True,
    ) -> pd.DataFrame:
        """Fit and transform with out-of-fold protection.
        
        This method computes target encodings using cross-validation to prevent
        data leakage. Each fold's encoding is computed using only the other folds.
        
        Args:
            df: Training DataFrame.
            cols: List of categorical columns to encode.
            target: Target values.
            n_folds: Number of folds for cross-validation.
            stratified: Whether to use stratified splitting.
            random_state: Random seed for reproducibility.
            return_full_df: If True, return full DataFrame with encoded columns.
                           If False, return only encoded columns.
        
        Returns:
            pd.DataFrame: DataFrame with target-encoded columns.
        """
        target = pd.Series(target).reset_index(drop=True)
        df = df.reset_index(drop=True)
        
        # Initialize result
        if return_full_df:
            result = df.copy()
        else:
            result = pd.DataFrame(index=df.index)
        
        # Create folds
        if stratified and len(np.unique(target)) > 1:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            split_iter = kf.split(df, target)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            split_iter = kf.split(df)
        
        # Process each column
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            encoded_values = np.zeros(len(df))
            
            # Compute OOF encodings
            for train_idx, valid_idx in split_iter:
                X_train, X_valid = df.iloc[train_idx], df.iloc[valid_idx]
                y_train = target.iloc[train_idx]
                
                # Compute encoding on training fold
                encoding = self._compute_encoding(X_train[col], y_train)
                
                # Apply encoding to validation fold
                valid_categories = X_valid[col].values
                for i, idx in enumerate(valid_idx):
                    cat = valid_categories[i]
                    if pd.isna(cat):
                        encoded_values[idx] = encoding["__global_mean__"]
                    else:
                        encoded_values[idx] = encoding.get(
                            cat, encoding["__global_mean__"]
                        )
            
            # Add noise if specified
            if self.noise > 0:
                encoded_values += np.random.normal(0, self.noise, len(encoded_values))
            
            # Store encoded column
            result[f"{col}_target_enc"] = encoded_values
            
            # Fit final encoding on full dataset for transform
            final_encoding = self._compute_encoding(df[col], target)
            self.encodings_[col] = final_encoding
            self.global_means_[col] = final_encoding["__global_mean__"]
        
        self.fitted_columns_ = cols
        self._is_fitted = True
        
        return result
    
    def transform(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        return_full_df: bool = True,
    ) -> pd.DataFrame:
        """Transform test data using fitted encodings.
        
        Args:
            df: Test DataFrame.
            cols: List of columns to encode. If None, uses fitted columns.
            return_full_df: If True, return full DataFrame with encoded columns.
        
        Returns:
            pd.DataFrame: DataFrame with target-encoded columns.
        
        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder must be fitted before transform. Call fit() or fit_transform() first.")
        
        cols = cols or self.fitted_columns_
        
        if return_full_df:
            result = df.copy()
        else:
            result = pd.DataFrame(index=df.index)
        
        for col in cols:
            if col not in df.columns:
                if self.handle_unknown == "error":
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                else:
                    warnings.warn(f"Column '{col}' not found. Skipping.")
                    continue
            
            if col not in self.encodings_:
                if self.handle_unknown == "error":
                    raise ValueError(f"Column '{col}' was not fitted")
                else:
                    warnings.warn(f"Column '{col}' was not fitted. Skipping.")
                    continue
            
            encoding = self.encodings_[col]
            global_mean = encoding["__global_mean__"]
            
            # Apply encoding
            def encode_value(x):
                if pd.isna(x):
                    return global_mean
                return encoding.get(x, global_mean)
            
            result[f"{col}_target_enc"] = df[col].apply(encode_value)
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get names of the encoded features.
        
        Returns:
            List of encoded feature names.
        """
        return [f"{col}_target_enc" for col in self.fitted_columns_]


class CountEncoder:
    """Count encoder for categorical variables.
    
    Encodes categorical variables by their frequency (count) in the training data.
    
    Args:
        normalize: Whether to normalize counts to proportions.
        min_count: Minimum count threshold. Categories with count below this
                  will be encoded as min_count or a special value.
        handle_unknown: Strategy for unknown categories ('value', 'zero', 'error').
    
    Example:
        >>> encoder = CountEncoder(normalize=True)
        >>> X_encoded = encoder.fit_transform(df, cols=['category'])
    """
    
    def __init__(
        self,
        normalize: bool = False,
        min_count: Optional[int] = None,
        handle_unknown: str = "value",
    ) -> None:
        """Initialize the CountEncoder.
        
        Args:
            normalize: Whether to normalize counts to proportions.
            min_count: Minimum count threshold.
            handle_unknown: Strategy for unknown categories.
        """
        self.normalize = normalize
        self.min_count = min_count
        self.handle_unknown = handle_unknown
        
        self.counts_: Dict[str, Dict] = {}
        self.fitted_columns_: List[str] = []
        self._is_fitted = False
    
    def fit(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> "CountEncoder":
        """Fit the encoder.
        
        Args:
            df: Training DataFrame.
            cols: List of columns to encode.
        
        Returns:
            CountEncoder: Fitted encoder.
        """
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            # Compute counts
            counts = df[col].value_counts(normalize=self.normalize)
            
            if self.min_count is not None and not self.normalize:
                # For non-normalized counts, apply threshold
                counts = counts[counts >= self.min_count]
            
            self.counts_[col] = counts.to_dict()
        
        self.fitted_columns_ = cols
        self._is_fitted = True
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Transform data using fitted counts.
        
        Args:
            df: DataFrame to transform.
            cols: Columns to encode.
        
        Returns:
            DataFrame with count-encoded columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder must be fitted before transform.")
        
        cols = cols or self.fitted_columns_
        result = df.copy()
        
        for col in cols:
            if col not in df.columns:
                continue
            
            counts = self.counts_.get(col, {})
            
            def encode_value(x):
                if pd.isna(x):
                    return 0
                if x in counts:
                    return counts[x]
                if self.handle_unknown == "zero":
                    return 0
                elif self.handle_unknown == "value":
                    return min(counts.values()) if counts else 0
                else:
                    raise ValueError(f"Unknown category: {x}")
            
            result[f"{col}_count_enc"] = df[col].apply(encode_value)
        
        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            df: Training DataFrame.
            cols: Columns to encode.
        
        Returns:
            DataFrame with count-encoded columns.
        """
        return self.fit(df, cols).transform(df, cols)


class FrequencyEncoder:
    """Frequency encoder that encodes categories by their frequency rank.
    
    This encoder replaces categories with their frequency rank (1 for most frequent).
    
    Args:
        ascending: If True, 1 is least frequent. If False, 1 is most frequent.
    
    Example:
        >>> encoder = FrequencyEncoder()
        >>> X_encoded = encoder.fit_transform(df, cols=['category'])
    """
    
    def __init__(self, ascending: bool = False) -> None:
        """Initialize the FrequencyEncoder.
        
        Args:
            ascending: Rank direction.
        """
        self.ascending = ascending
        self.ranks_: Dict[str, Dict] = {}
        self.fitted_columns_: List[str] = []
        self._is_fitted = False
    
    def fit(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> "FrequencyEncoder":
        """Fit the encoder.
        
        Args:
            df: Training DataFrame.
            cols: Columns to encode.
        
        Returns:
            FrequencyEncoder: Fitted encoder.
        """
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            # Compute frequency ranks
            counts = df[col].value_counts(ascending=self.ascending)
            ranks = {cat: rank for rank, cat in enumerate(counts.index, 1)}
            self.ranks_[col] = ranks
        
        self.fitted_columns_ = cols
        self._is_fitted = True
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Transform data using fitted ranks.
        
        Args:
            df: DataFrame to transform.
            cols: Columns to encode.
        
        Returns:
            DataFrame with frequency-encoded columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder must be fitted before transform.")
        
        cols = cols or self.fitted_columns_
        result = df.copy()
        
        for col in cols:
            if col not in df.columns:
                continue
            
            ranks = self.ranks_.get(col, {})
            max_rank = max(ranks.values()) if ranks else 0
            
            result[f"{col}_freq_enc"] = df[col].map(ranks).fillna(max_rank + 1)
        
        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            df: Training DataFrame.
            cols: Columns to encode.
        
        Returns:
            DataFrame with frequency-encoded columns.
        """
        return self.fit(df, cols).transform(df, cols)
