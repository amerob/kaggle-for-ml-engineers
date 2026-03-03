"""
Groupby feature engineering for ML competitions.

This module provides functions to create aggregation features based on groupby
operations, including difference and ratio features, hierarchical groupbys,
and GPU-accelerated versions using cuDF.

Author: Amer Hussein
"""

import warnings
from typing import List, Dict, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd


def create_groupby_features(
    df: pd.DataFrame,
    groupby_columns: Union[str, List[str]],
    numerical_columns: Optional[List[str]] = None,
    aggregations: List[str] = None,
    diff_features: bool = True,
    ratio_features: bool = True,
    hierarchical: bool = False,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    """Create groupby aggregation features.
    
    This function creates aggregation features by grouping data and computing
    statistics like mean, std, min, max, etc. It also creates difference and
    ratio features comparing individual values to group statistics.
    
    Args:
        df: Input DataFrame.
        groupby_columns: Column(s) to group by. Can be a single column name
                        or a list for multi-column grouping.
        numerical_columns: Numerical columns to aggregate. If None, uses all
                          numerical columns except groupby columns.
        aggregations: List of aggregation functions. Default includes common
                     aggregations: ['mean', 'std', 'min', 'max', 'median', 'count', 'nunique'].
        diff_features: Whether to create difference features (value - group_mean).
        ratio_features: Whether to create ratio features (value / group_mean).
        hierarchical: Whether to create hierarchical groupby features for
                     multi-column grouping.
        prefix: Optional prefix for new column names.
        suffix: Optional suffix for new column names.
    
    Returns:
        pd.DataFrame: DataFrame with additional groupby features.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'category': ['A', 'A', 'B', 'B'],
        ...     'value': [1, 2, 3, 4]
        ... })
        >>> df_features = create_groupby_features(
        ...     df,
        ...     groupby_columns='category',
        ...     numerical_columns=['value'],
        ...     aggregations=['mean', 'std']
        ... )
        >>> # Creates: value_mean_by_category, value_std_by_category,
        >>> #          value_diff_from_mean_by_category, value_ratio_to_mean_by_category
    """
    # Validate inputs
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    
    for col in groupby_columns:
        if col not in df.columns:
            raise ValueError(f"Groupby column '{col}' not found in DataFrame")
    
    # Default aggregations
    if aggregations is None:
        aggregations = ["mean", "std", "min", "max", "median", "count", "nunique"]
    
    # Auto-detect numerical columns if not specified
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        numerical_columns = [c for c in numerical_columns if c not in groupby_columns]
    
    # Filter to existing columns
    numerical_columns = [c for c in numerical_columns if c in df.columns]
    
    if len(numerical_columns) == 0:
        warnings.warn("No numerical columns found for aggregation")
        return df.copy()
    
    result = df.copy()
    
    # Create groupby name for column naming
    groupby_name = "_".join(groupby_columns)
    if prefix:
        groupby_name = f"{prefix}_{groupby_name}"
    if suffix:
        groupby_name = f"{groupby_name}_{suffix}"
    
    # Compute aggregations
    agg_dict = {col: aggregations for col in numerical_columns}
    
    try:
        grouped = df.groupby(groupby_columns).agg(agg_dict)
        
        # Flatten column names
        grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
        
        # Rename with groupby prefix
        grouped = grouped.rename(columns={
            col: f"{col}_by_{groupby_name}" 
            for col in grouped.columns
        })
        
        # Merge back to original dataframe
        result = result.merge(
            grouped.reset_index(),
            on=groupby_columns,
            how="left"
        )
        
        # Create difference features
        if diff_features:
            for col in numerical_columns:
                mean_col = f"{col}_mean_by_{groupby_name}"
                if mean_col in result.columns:
                    result[f"{col}_diff_from_mean_by_{groupby_name}"] = (
                        result[col] - result[mean_col]
                    )
                    
                    # Also create median diff if available
                    median_col = f"{col}_median_by_{groupby_name}"
                    if median_col in result.columns:
                        result[f"{col}_diff_from_median_by_{groupby_name}"] = (
                            result[col] - result[median_col]
                        )
        
        # Create ratio features
        if ratio_features:
            for col in numerical_columns:
                mean_col = f"{col}_mean_by_{groupby_name}"
                if mean_col in result.columns:
                    # Avoid division by zero
                    result[f"{col}_ratio_to_mean_by_{groupby_name}"] = (
                        result[col] / (result[mean_col] + 1e-8)
                    )
                    
                    # Also create median ratio if available
                    median_col = f"{col}_median_by_{groupby_name}"
                    if median_col in result.columns:
                        result[f"{col}_ratio_to_median_by_{groupby_name}"] = (
                            result[col] / (result[median_col] + 1e-8)
                        )
        
        # Create hierarchical features if requested
        if hierarchical and len(groupby_columns) > 1:
            for i in range(1, len(groupby_columns)):
                sub_groupby = groupby_columns[:i + 1]
                sub_name = "_".join(sub_groupby)
                if prefix:
                    sub_name = f"{prefix}_{sub_name}"
                if suffix:
                    sub_name = f"{sub_name}_{suffix}"
                
                sub_grouped = df.groupby(sub_groupby)[numerical_columns].mean()
                sub_grouped = sub_grouped.rename(columns={
                    col: f"{col}_mean_by_{sub_name}"
                    for col in numerical_columns
                })
                
                result = result.merge(
                    sub_grouped.reset_index(),
                    on=sub_groupby,
                    how="left"
                )
    
    except Exception as e:
        warnings.warn(f"Error creating groupby features: {e}")
    
    return result


def create_groupby_features_gpu(
    df: pd.DataFrame,
    groupby_columns: Union[str, List[str]],
    numerical_columns: Optional[List[str]] = None,
    aggregations: List[str] = None,
    diff_features: bool = True,
    ratio_features: bool = True,
) -> pd.DataFrame:
    """GPU-accelerated version of create_groupby_features using cuDF.
    
    This function provides the same functionality as create_groupby_features
    but uses cuDF for GPU acceleration. Falls back to CPU if cuDF is not available.
    
    Args:
        df: Input DataFrame (pandas or cuDF).
        groupby_columns: Column(s) to group by.
        numerical_columns: Numerical columns to aggregate.
        aggregations: List of aggregation functions.
        diff_features: Whether to create difference features.
        ratio_features: Whether to create ratio features.
    
    Returns:
        pd.DataFrame: DataFrame with additional groupby features.
    
    Example:
        >>> df_features = create_groupby_features_gpu(
        ...     df,
        ...     groupby_columns='category',
        ...     numerical_columns=['value']
        ... )
    """
    try:
        import cudf
        from cuml.preprocessing import StandardScaler
        CUDF_AVAILABLE = True
    except ImportError:
        CUDF_AVAILABLE = False
        warnings.warn("cuDF not available. Falling back to CPU implementation.")
        return create_groupby_features(
            df, groupby_columns, numerical_columns, aggregations,
            diff_features, ratio_features
        )
    
    # Convert to cuDF if needed
    if isinstance(df, pd.DataFrame):
        df_cudf = cudf.from_pandas(df)
    else:
        df_cudf = df
    
    # Validate inputs
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    
    # Default aggregations
    if aggregations is None:
        aggregations = ["mean", "std", "min", "max"]
    
    # Auto-detect numerical columns
    if numerical_columns is None:
        numerical_columns = df_cudf.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        numerical_columns = [c for c in numerical_columns if c not in groupby_columns]
    
    result = df_cudf.copy()
    groupby_name = "_".join(groupby_columns)
    
    # Compute aggregations using cuDF
    for col in numerical_columns:
        if col not in df_cudf.columns:
            continue
        
        for agg in aggregations:
            if hasattr(df_cudf.groupby(groupby_columns)[col], agg):
                agg_result = getattr(df_cudf.groupby(groupby_columns)[col], agg)()
                agg_result = agg_result.reset_index()
                agg_result.columns = groupby_columns + [f"{col}_{agg}_by_{groupby_name}"]
                result = result.merge(agg_result, on=groupby_columns, how="left")
    
    # Create difference and ratio features
    if diff_features or ratio_features:
        for col in numerical_columns:
            mean_col = f"{col}_mean_by_{groupby_name}"
            if mean_col in result.columns:
                if diff_features:
                    result[f"{col}_diff_from_mean_by_{groupby_name}"] = (
                        result[col] - result[mean_col]
                    )
                if ratio_features:
                    result[f"{col}_ratio_to_mean_by_{groupby_name}"] = (
                        result[col] / (result[mean_col] + 1e-8)
                    )
    
    # Convert back to pandas
    return result.to_pandas()


def create_rolling_groupby_features(
    df: pd.DataFrame,
    groupby_columns: Union[str, List[str]],
    numerical_columns: List[str],
    time_column: str,
    windows: List[int] = None,
    aggregations: List[str] = None,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Create rolling window groupby features for time series data.
    
    Args:
        df: Input DataFrame.
        groupby_columns: Column(s) to group by.
        numerical_columns: Numerical columns to aggregate.
        time_column: Column containing timestamps.
        windows: List of window sizes (in number of observations).
        aggregations: List of aggregation functions.
        min_periods: Minimum number of observations required.
    
    Returns:
        pd.DataFrame: DataFrame with rolling groupby features.
    
    Example:
        >>> df_features = create_rolling_groupby_features(
        ...     df,
        ...     groupby_columns='store_id',
        ...     numerical_columns=['sales'],
        ...     time_column='date',
        ...     windows=[7, 14, 30]
        ... )
    """
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    
    if windows is None:
        windows = [7, 14, 30]
    
    if aggregations is None:
        aggregations = ["mean", "std", "sum"]
    
    result = df.copy()
    groupby_name = "_".join(groupby_columns)
    
    # Sort by time within each group
    df_sorted = df.sort_values(by=groupby_columns + [time_column])
    
    for col in numerical_columns:
        if col not in df.columns:
            continue
        
        for window in windows:
            for agg in aggregations:
                col_name = f"{col}_rolling{window}_{agg}_by_{groupby_name}"
                
                if agg == "mean":
                    result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=min_periods).mean()
                    )
                elif agg == "std":
                    result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=min_periods).std()
                    )
                elif agg == "sum":
                    result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=min_periods).sum()
                    )
                elif agg == "min":
                    result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=min_periods).min()
                    )
                elif agg == "max":
                    result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=min_periods).max()
                    )
    
    return result


def create_expanding_groupby_features(
    df: pd.DataFrame,
    groupby_columns: Union[str, List[str]],
    numerical_columns: List[str],
    time_column: str,
    aggregations: List[str] = None,
) -> pd.DataFrame:
    """Create expanding window groupby features for time series data.
    
    Expanding windows include all previous observations in each group.
    
    Args:
        df: Input DataFrame.
        groupby_columns: Column(s) to group by.
        numerical_columns: Numerical columns to aggregate.
        time_column: Column containing timestamps.
        aggregations: List of aggregation functions.
    
    Returns:
        pd.DataFrame: DataFrame with expanding groupby features.
    
    Example:
        >>> df_features = create_expanding_groupby_features(
        ...     df,
        ...     groupby_columns='store_id',
        ...     numerical_columns=['sales'],
        ...     time_column='date'
        ... )
    """
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    
    if aggregations is None:
        aggregations = ["mean", "std", "sum", "count"]
    
    result = df.copy()
    groupby_name = "_".join(groupby_columns)
    
    # Sort by time within each group
    df_sorted = df.sort_values(by=groupby_columns + [time_column])
    
    for col in numerical_columns:
        if col not in df.columns:
            continue
        
        for agg in aggregations:
            col_name = f"{col}_expanding_{agg}_by_{groupby_name}"
            
            if agg == "mean":
                result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                    lambda x: x.expanding().mean()
                )
            elif agg == "std":
                result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                    lambda x: x.expanding().std()
                )
            elif agg == "sum":
                result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                    lambda x: x.expanding().sum()
                )
            elif agg == "count":
                result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                    lambda x: x.expanding().count()
                )
            elif agg == "min":
                result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                    lambda x: x.expanding().min()
                )
            elif agg == "max":
                result[col_name] = df_sorted.groupby(groupby_columns)[col].transform(
                    lambda x: x.expanding().max()
                )
    
    return result


def create_lag_features(
    df: pd.DataFrame,
    groupby_columns: Union[str, List[str]],
    numerical_columns: List[str],
    time_column: str,
    lags: List[int] = None,
) -> pd.DataFrame:
    """Create lag features grouped by specified columns.
    
    Args:
        df: Input DataFrame.
        groupby_columns: Column(s) to group by.
        numerical_columns: Numerical columns to create lags for.
        time_column: Column containing timestamps.
        lags: List of lag periods.
    
    Returns:
        pd.DataFrame: DataFrame with lag features.
    
    Example:
        >>> df_features = create_lag_features(
        ...     df,
        ...     groupby_columns='store_id',
        ...     numerical_columns=['sales'],
        ...     time_column='date',
        ...     lags=[1, 7, 14]
        ... )
    """
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    
    if lags is None:
        lags = [1, 7, 14]
    
    result = df.copy()
    groupby_name = "_".join(groupby_columns)
    
    # Sort by time within each group
    df_sorted = df.sort_values(by=groupby_columns + [time_column])
    
    for col in numerical_columns:
        if col not in df.columns:
            continue
        
        for lag in lags:
            col_name = f"{col}_lag{lag}_by_{groupby_name}"
            result[col_name] = df_sorted.groupby(groupby_columns)[col].shift(lag)
    
    return result


def create_target_encoded_groupby(
    df: pd.DataFrame,
    groupby_columns: Union[str, List[str]],
    target_column: str,
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """Create target-encoded groupby features with smoothing.
    
    Args:
        df: Input DataFrame.
        groupby_columns: Column(s) to group by.
        target_column: Target column to encode.
        smoothing: Smoothing parameter for regularization.
    
    Returns:
        pd.DataFrame: DataFrame with target-encoded groupby features.
    
    Example:
        >>> df_features = create_target_encoded_groupby(
        ...     df,
        ...     groupby_columns=['category', 'subcategory'],
        ...     target_column='target',
        ...     smoothing=10.0
        ... )
    """
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    
    result = df.copy()
    groupby_name = "_".join(groupby_columns)
    
    # Compute global mean
    global_mean = df[target_column].mean()
    
    # Compute group statistics
    stats = df.groupby(groupby_columns)[target_column].agg(["count", "mean"])
    
    # Apply smoothing
    smoothed_means = (
        stats["count"] * stats["mean"] + smoothing * global_mean
    ) / (stats["count"] + smoothing)
    
    # Create feature name
    feature_name = f"{target_column}_encoded_by_{groupby_name}"
    
    # Map to dataframe
    encoding_dict = smoothed_means.to_dict()
    result[feature_name] = result.apply(
        lambda row: encoding_dict.get(
            tuple(row[col] for col in groupby_columns), global_mean
        ),
        axis=1
    )
    
    return result
