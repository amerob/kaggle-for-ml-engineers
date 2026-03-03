# Part 3: Feature Engineering

## The Foundation of Competitive Success

Feature engineering remains the highest-leverage activity in competitive machine learning. While model architectures get the headlines, well-engineered features often provide 2-5x the improvement of hyperparameter tuning.

### Feature Engineering Impact Hierarchy

```
Impact
    │
    │    ╭─────── Domain Features
    │   ╱
    │  ╱    ╭─────── Interaction Features
    │ ╱    ╱
    │╱    ╱    ╭─────── Target Encoding
    ├────╱────╱
    │   /    /
    │  /    /    ╭─────── Hyperparameter Tuning
    │ /    /    /
    │/____/____/____________
     Low          High
              Effort Required
```

---

## Safe Target Encoding with Out-of-Fold (OOF)

Target encoding is powerful but dangerous—naive implementation causes severe overfitting. The solution is out-of-fold (OOF) encoding.

### The Problem with Naive Target Encoding

```python
# DANGEROUS: Leaks target information
from sklearn.preprocessing import TargetEncoder

# This uses the target from the same row being encoded!
encoder = TargetEncoder()
df['category_encoded'] = encoder.fit_transform(
    df[['category']], df['target']
)
```

### SafeTargetEncoder Implementation

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SafeTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Safe target encoder using out-of-fold encoding.
    
    Prevents target leakage by encoding each fold using 
    statistics from other folds only.
    """
    
    def __init__(self, columns, smoothing=10, min_samples_leaf=1, noise=0):
        """
        Args:
            columns: List of categorical columns to encode
            smoothing: Smoothing factor for mean encoding
            min_samples_leaf: Minimum samples for category
            noise: Amount of Gaussian noise to add (0 for none)
        """
        self.columns = columns if isinstance(columns, list) else [columns]
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise = noise
        self.encoding_maps = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """
        Fit encoder on training data.
        
        Note: This computes global statistics only.
        Actual encoding happens per-fold in transform.
        """
        self.global_mean = np.mean(y)
        
        for col in self.columns:
            # Compute global statistics per category
            stats = pd.DataFrame({
                'target': y,
                col: X[col]
            }).groupby(col)['target'].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothed_mean = (
                (stats['count'] * stats['mean'] + 
                 self.smoothing * self.global_mean) /
                (stats['count'] + self.smoothing)
            )
            
            self.encoding_maps[col] = smoothed_mean.to_dict()
            
        return self
    
    def transform(self, X, y=None):
        """
        Transform data using learned encodings.
        
        For training data (y provided), use OOF encoding.
        For test data (y=None), use fitted encodings.
        """
        X_encoded = X.copy()
        
        for col in self.columns:
            if y is not None:
                # Training: Use OOF encoding
                X_encoded[col] = self._oof_encode(X, y, col)
            else:
                # Test: Use fitted encodings
                X_encoded[col] = X[col].map(
                    self.encoding_maps[col]
                ).fillna(self.global_mean)
                
        return X_encoded
    
    def _oof_encode(self, X, y, col):
        """
        Perform out-of-fold target encoding.
        
        Each row is encoded using target statistics from 
        all other rows with the same category.
        """
        from sklearn.model_selection import KFold
        
        oof_encoded = np.zeros(len(X))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            # Compute encoding from training folds
            train_data = X.iloc[train_idx]
            train_y = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            
            fold_stats = pd.DataFrame({
                'target': train_y,
                col: train_data[col]
            }).groupby(col)['target'].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothed_mean = (
                (fold_stats['count'] * fold_stats['mean'] + 
                 self.smoothing * self.global_mean) /
                (fold_stats['count'] + self.smoothing)
            )
            
            # Encode validation fold
            val_data = X.iloc[val_idx]
            oof_encoded[val_idx] = val_data[col].map(
                smoothed_mean.to_dict()
            ).fillna(self.global_mean).values
            
        # Add noise if specified
        if self.noise > 0:
            oof_encoded += np.random.normal(0, self.noise, len(oof_encoded))
            
        return oof_encoded
    
    def fit_transform(self, X, y):
        """Fit and transform with OOF encoding."""
        self.fit(X, y)
        return self.transform(X, y)
```

### Target Encoding Best Practices

| Parameter | Default | When to Increase | When to Decrease |
|-----------|---------|------------------|------------------|
| `smoothing` | 10 | High cardinality, noisy data | Low cardinality, clean data |
| `min_samples_leaf` | 1 | Rare categories | All categories well-represented |
| `noise` | 0 | High overfitting risk | Stable validation |

---

## Groupby Feature Engineering

Groupby aggregations are among the most powerful feature engineering techniques for tabular data.

### Comprehensive Groupby Feature Generator

```python
def create_groupby_features(df, group_cols, agg_cols, 
                            aggregations=None, prefix=None):
    """
    Create comprehensive groupby aggregation features.
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by (list or single)
        agg_cols: Columns to aggregate (list or single)
        aggregations: List of aggregation functions
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with new features
    """
    if aggregations is None:
        aggregations = ['mean', 'std', 'min', 'max', 'count', 
                       'median', 'skew', 'nunique']
    
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(agg_cols, str):
        agg_cols = [agg_cols]
        
    if prefix is None:
        prefix = '_'.join(group_cols)
    
    result = df.copy()
    
    for agg_col in agg_cols:
        # Skip non-numeric columns
        if not np.issubdtype(df[agg_col].dtype, np.number):
            continue
            
        # Compute aggregations
        agg_dict = {agg: agg for agg in aggregations 
                   if agg in ['mean', 'std', 'min', 'max', 
                             'count', 'median', 'skew', 'sum']}
        
        grouped = df.groupby(group_cols)[agg_col].agg(agg_dict)
        
        # Rename columns
        grouped.columns = [
            f'{prefix}_{agg_col}_{agg}' 
            for agg in grouped.columns
        ]
        
        # Merge back
        result = result.merge(
            grouped, 
            left_on=group_cols, 
            right_index=True, 
            how='left'
        )
        
        # Add ratio features
        mean_col = f'{prefix}_{agg_col}_mean'
        if mean_col in result.columns:
            result[f'{agg_col}_to_{prefix}_mean_ratio'] = (
                result[agg_col] / (result[mean_col] + 1e-8)
            )
            result[f'{agg_col}_minus_{prefix}_mean'] = (
                result[agg_col] - result[mean_col]
            )
    
    return result


# Example usage
# df = create_groupby_features(
#     df, 
#     group_cols=['customer_id', 'product_category'],
#     agg_cols=['purchase_amount', 'quantity'],
#     aggregations=['mean', 'std', 'min', 'max', 'count']
# )
```

### Advanced Groupby Patterns

```python
def create_time_based_groupby_features(df, group_col, time_col, 
                                       value_col, windows=[7, 30, 90]):
    """
    Create time-windowed groupby features.
    
    Useful for customer behavior, sales forecasting, etc.
    """
    df = df.sort_values([group_col, time_col])
    
    for window in windows:
        # Rolling window aggregations within each group
        df[f'{value_col}_{window}d_mean'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        
        df[f'{value_col}_{window}d_std'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        
        df[f'{value_col}_{window}d_max'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(window, min_periods=1).max())
        )
        
        # Days since last transaction
        df[f'days_since_last_{value_col}'] = (
            df.groupby(group_col)[time_col]
            .diff().dt.days
        )
    
    return df
```

---

## OOF Pipelines: The Complete Pattern

Out-of-fold processing is the gold standard for preventing leakage in feature engineering.

### Complete OOF Pipeline Implementation

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

class OOFPipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline for out-of-fold feature engineering.
    
    Ensures all feature engineering respects fold boundaries,
    preventing any leakage from validation to training.
    """
    
    def __init__(self, transformers, n_splits=5):
        """
        Args:
            transformers: List of (name, transformer) tuples
            n_splits: Number of CV folds for OOF processing
        """
        self.transformers = transformers
        self.n_splits = n_splits
        self.fitted_transformers = []
        
    def fit_transform(self, X, y):
        """
        Fit and transform with OOF processing.
        
        Each fold is transformed using transformers fitted on other folds.
        """
        X_transformed = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Store OOF predictions
        oof_features = np.zeros((len(X), len(self.transformers)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            
            fold_transformers = []
            
            for i, (name, transformer) in enumerate(self.transformers):
                # Fit on training fold
                transformer_copy = clone(transformer)
                
                if hasattr(transformer_copy, 'fit_transform'):
                    # Target-aware transformer (e.g., target encoder)
                    train_transformed = transformer_copy.fit_transform(
                        X_train, y_train
                    )
                    val_transformed = transformer_copy.transform(X_val)
                else:
                    # Target-agnostic transformer
                    transformer_copy.fit(X_train)
                    train_transformed = transformer_copy.transform(X_train)
                    val_transformed = transformer_copy.transform(X_val)
                
                fold_transformers.append(transformer_copy)
                
                # Store OOF features
                if isinstance(val_transformed, pd.DataFrame):
                    oof_features[val_idx, i] = val_transformed.iloc[:, 0].values
                else:
                    oof_features[val_idx, i] = val_transformed[:, 0]
            
            self.fitted_transformers.append(fold_transformers)
        
        # Add OOF features to dataframe
        for i, (name, _) in enumerate(self.transformers):
            X_transformed[f'oof_{name}'] = oof_features[:, i]
        
        return X_transformed
    
    def transform(self, X):
        """
        Transform test data using all fitted transformers.
        
        Averages predictions from all fold transformers.
        """
        X_transformed = X.copy()
        
        for i, (name, _) in enumerate(self.transformers):
            fold_predictions = []
            
            for fold_transformers in self.fitted_transformers:
                transformer = fold_transformers[i]
                pred = transformer.transform(X)
                
                if isinstance(pred, pd.DataFrame):
                    fold_predictions.append(pred.iloc[:, 0].values)
                else:
                    fold_predictions.append(pred[:, 0])
            
            # Average across folds
            X_transformed[f'oof_{name}'] = np.mean(fold_predictions, axis=0)
        
        return X_transformed


def clone(estimator):
    """Clone a scikit-learn estimator."""
    from sklearn.base import clone as sk_clone
    return sk_clone(estimator)
```

---

## Automated Feature Engineering

### FeatureTools Integration

```python
import featuretools as ft

def create_automated_features(df, entity_id, target_entity, 
                              max_depth=2, verbose=True):
    """
    Create automated features using FeatureTools.
    
    Args:
        df: Input DataFrame
        entity_id: Name for the entity
        target_entity: Entity to create features for
        max_depth: Maximum depth of feature stacking
        verbose: Print progress
        
    Returns:
        DataFrame with automated features
    """
    # Create entity set
    es = ft.EntitySet(id="competition_data")
    
    # Add entity
    es = es.add_dataframe(
        dataframe_name=entity_id,
        dataframe=df,
        index=df.index.name or 'id',
        make_index=True if df.index.name is None else False
    )
    
    # Run deep feature synthesis
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name=target_entity,
        agg_primitives=["mean", "max", "min", "std", "count", 
                       "sum", "skew", "mode"],
        trans_primitives=["diff", "absolute", "add_numeric", 
                         "multiply_numeric"],
        max_depth=max_depth,
        verbose=verbose
    )
    
    if verbose:
        print(f"Created {len(feature_defs)} features")
        
    return feature_matrix, feature_defs
```

---

## GPU-Accelerated Feature Engineering

### cuDF Implementation

```python
def create_features_cudf(df, group_cols, agg_cols):
    """
    Create groupby features using GPU acceleration via cuDF.
    
    Requires: pip install cudf-cu11 (or appropriate CUDA version)
    
    Args:
        df: pandas DataFrame
        group_cols: Columns to group by
        agg_cols: Columns to aggregate
        
    Returns:
        pandas DataFrame with new features
    """
    import cudf
    
    # Convert to cuDF
    gdf = cudf.DataFrame.from_pandas(df)
    
    # Perform aggregations on GPU
    for agg_col in agg_cols:
        # Mean
        mean_values = gdf.groupby(group_cols)[agg_col].mean()
        
        # Merge back
        gdf = gdf.merge(
            mean_values.rename(f'{agg_col}_mean'),
            left_on=group_cols,
            right_index=True,
            how='left'
        )
        
        # Standard deviation
        std_values = gdf.groupby(group_cols)[agg_col].std()
        gdf = gdf.merge(
            std_values.rename(f'{agg_col}_std'),
            left_on=group_cols,
            right_index=True,
            how='left'
        )
    
    # Convert back to pandas
    return gdf.to_pandas()


# Performance comparison helper
def benchmark_feature_engineering(df, group_cols, agg_cols):
    """
    Benchmark CPU vs GPU feature engineering.
    """
    import time
    
    # CPU benchmark
    start = time.time()
    cpu_result = create_groupby_features(df, group_cols, agg_cols)
    cpu_time = time.time() - start
    
    # GPU benchmark
    start = time.time()
    gpu_result = create_features_cudf(df, group_cols, agg_cols)
    gpu_time = time.time() - start
    
    print(f"CPU time: {cpu_time:.2f}s")
    print(f"GPU time: {gpu_time:.2f}s")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
    return cpu_result, gpu_result
```

---

## Feature Selection Pipeline

### Multi-Method Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FeatureSelector:
    """
    Multi-method feature selection pipeline.
    
    Combines multiple selection strategies for robust feature selection.
    """
    
    def __init__(self, n_features=50):
        self.n_features = n_features
        self.selected_features = None
        self.feature_scores = {}
        
    def fit(self, X, y):
        """
        Fit feature selector using multiple methods.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        feature_names = X.columns.tolist()
        
        # Method 1: Univariate statistical tests
        selector_f = SelectKBest(f_classif, k='all')
        selector_f.fit(X, y)
        self.feature_scores['f_classif'] = dict(
            zip(feature_names, selector_f.scores_)
        )
        
        # Method 2: Mutual information
        selector_mi = SelectKBest(mutual_info_classif, k='all')
        selector_mi.fit(X, y)
        self.feature_scores['mutual_info'] = dict(
            zip(feature_names, selector_mi.scores_)
        )
        
        # Method 3: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        self.feature_scores['rf_importance'] = dict(
            zip(feature_names, rf.feature_importances_)
        )
        
        # Method 4: Recursive Feature Elimination
        rfe = RFE(
            RandomForestClassifier(n_estimators=50, random_state=42),
            n_features_to_select=self.n_features
        )
        rfe.fit(X, y)
        self.feature_scores['rfe_selected'] = dict(
            zip(feature_names, rfe.support_)
        )
        
        # Combine scores (voting approach)
        combined_scores = self._combine_scores(feature_names)
        
        # Select top features
        self.selected_features = [
            f for f, s in sorted(
                combined_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.n_features]
        ]
        
        return self
    
    def _combine_scores(self, feature_names):
        """Combine scores from multiple methods."""
        combined = {}
        
        for feature in feature_names:
            votes = 0
            
            # Vote from f_classif (top 50%)
            f_scores = self.feature_scores['f_classif']
            threshold = np.percentile(list(f_scores.values()), 50)
            if f_scores[feature] > threshold:
                votes += 1
            
            # Vote from mutual info (top 50%)
            mi_scores = self.feature_scores['mutual_info']
            threshold = np.percentile(list(mi_scores.values()), 50)
            if mi_scores[feature] > threshold:
                votes += 1
            
            # Vote from RF importance (top 50%)
            rf_scores = self.feature_scores['rf_importance']
            threshold = np.percentile(list(rf_scores.values()), 50)
            if rf_scores[feature] > threshold:
                votes += 1
            
            # Vote from RFE
            if self.feature_scores['rfe_selected'][feature]:
                votes += 1
            
            combined[feature] = votes
        
        return combined
    
    def transform(self, X):
        """Transform data to selected features."""
        return X[self.selected_features]
    
    def fit_transform(self, X, y):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance_report(self):
        """Get detailed feature importance report."""
        report = []
        
        for feature in self.selected_features:
            report.append({
                'feature': feature,
                'f_classif_rank': self._get_rank('f_classif', feature),
                'mutual_info_rank': self._get_rank('mutual_info', feature),
                'rf_importance_rank': self._get_rank('rf_importance', feature),
                'rfe_selected': self.feature_scores['rfe_selected'][feature],
                'combined_votes': self._combine_scores([feature])[feature]
            })
        
        return pd.DataFrame(report)
    
    def _get_rank(self, method, feature):
        """Get rank of feature for a given method."""
        scores = self.feature_scores[method]
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (f, _) in enumerate(sorted_features, 1):
            if f == feature:
                return rank
        return None
```

---

## Production-Ready Feature Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

class ProductionFeaturePipeline:
    """
    Production-ready feature engineering pipeline.
    
    Separates online and offline feature computation,
    with versioning and monitoring support.
    """
    
    def __init__(self):
        self.offline_pipeline = None
        self.online_pipeline = None
        self.feature_metadata = {}
        self.version = '1.0.0'
        
    def build_pipeline(self, numeric_features, categorical_features, 
                       target_encoder_cols=None):
        """
        Build complete feature pipeline.
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            target_encoder_cols: Columns for target encoding
        """
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        self.offline_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        
        # Store metadata
        self.feature_metadata = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'target_encoder_cols': target_encoder_cols or [],
            'version': self.version,
            'n_features': len(numeric_features) + len(categorical_features)
        }
        
        return self
    
    def fit_offline(self, X, y=None):
        """
        Fit offline pipeline on training data.
        
        This is done once during training.
        """
        self.offline_pipeline.fit(X, y)
        return self
    
    def transform_offline(self, X):
        """Transform batch data (training or batch inference)."""
        return self.offline_pipeline.transform(X)
    
    def transform_online(self, X):
        """
        Transform single row for online inference.
        
        Optimized for low latency.
        """
        # Simplified transformation for online serving
        # In production, this would use pre-computed statistics
        return self.offline_pipeline.transform(X)
    
    def save(self, path):
        """Save pipeline to disk."""
        joblib.dump({
            'offline_pipeline': self.offline_pipeline,
            'online_pipeline': self.online_pipeline,
            'feature_metadata': self.feature_metadata,
            'version': self.version
        }, path)
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load pipeline from disk."""
        data = joblib.load(path)
        
        pipeline = cls()
        pipeline.offline_pipeline = data['offline_pipeline']
        pipeline.online_pipeline = data['online_pipeline']
        pipeline.feature_metadata = data['feature_metadata']
        pipeline.version = data['version']
        
        print(f"Pipeline loaded (version {pipeline.version})")
        return pipeline
    
    def get_feature_names(self):
        """Get output feature names."""
        if self.offline_pipeline is None:
            return []
        
        # Extract feature names from preprocessor
        preprocessor = self.offline_pipeline.named_steps['preprocessor']
        
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # Get one-hot encoded feature names
                cat_features = transformer.get_feature_names_out(columns)
                feature_names.extend(cat_features)
        
        return feature_names
```

---

## Feature Engineering Checklist

- [ ] Analyze data types and distributions
- [ ] Identify group structures for aggregations
- [ ] Implement safe target encoding with OOF
- [ ] Create domain-specific features
- [ ] Generate interaction features
- [ ] Add time-based features (if applicable)
- [ ] Run automated feature engineering (FeatureTools)
- [ ] Apply feature selection (remove low-importance features)
- [ ] Validate features don't cause leakage
- [ ] Profile feature computation time
- [ ] Document feature rationale
- [ ] Version feature pipeline

---

## Key Takeaways

1. **Feature engineering is the highest-leverage activity**—invest time here first
2. **Always use OOF encoding** for target-based features
3. **Groupby aggregations are powerful**—master the patterns
4. **Automate where possible**—FeatureTools for exploration, custom code for production
5. **GPU acceleration helps**—cuDF for large datasets
6. **Select features systematically**—multiple methods, voting approach
7. **Separate online/offline pipelines**—critical for production
