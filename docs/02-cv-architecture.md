# Part 2: Mental Models & Cross-Validation Architecture

## Leaderboard Dynamics: Understanding the Game

Success in Kaggle competitions requires understanding how the leaderboard works, what drives shake-ups, and how to build validation strategies that correlate with final rankings.

### The LB Signal Model

Think of the leaderboard as a noisy signal of true model performance:

```
True Performance = Public LB Score + Noise + Bias
```

Where:
- **Noise**: Random variation due to small test set size
- **Bias**: Systematic differences between public and private test sets

### Test Set Size Impact

| Test Set Size | Standard Error | Reliability |
|---------------|----------------|-------------|
| < 1,000       | ±0.03 (3%)     | Very Low    |
| 1,000 - 5,000 | ±0.015 (1.5%)  | Low         |
| 5,000 - 20,000| ±0.008 (0.8%)  | Medium      |
| 20,000 - 100k | ±0.004 (0.4%)  | High        |
| > 100,000     | ±0.002 (0.2%)  | Very High   |

### Standard Error Formula

```python
import numpy as np

def calculate_standard_error(auc_score, n_positive, n_negative):
    """
    Calculate standard error for AUC metric.
    
    Based on Hanley-McNeil formula.
    
    Args:
        auc_score: Observed AUC score
        n_positive: Number of positive samples
        n_negative: Number of negative samples
        
    Returns:
        Standard error of AUC
    """
    q1 = auc_score / (2 - auc_score)
    q2 = 2 * auc_score**2 / (1 + auc_score)
    
    se = np.sqrt(
        (auc_score * (1 - auc_score) + 
         (n_positive - 1) * (q1 - auc_score**2) + 
         (n_negative - 1) * (q2 - auc_score**2)) / 
        (n_positive * n_negative)
    )
    return se

# Example usage
auc = 0.85
n_pos = 1000
n_neg = 9000
se = calculate_standard_error(auc, n_pos, n_neg)
print(f"AUC: {auc:.4f} ± {se:.4f}")
```

---

## Shake-up Analysis

### What Causes Shake-ups?

Shake-ups occur when rankings change significantly between public and private leaderboards. Main causes:

1. **Distribution Shift**: Public and private sets come from different distributions
2. **Temporal Split**: Time-based splits create non-stationarity
3. **Small Sample Size**: High variance in small test sets
4. **Overfitting**: Models optimized for public LB fail to generalize

### Shake-up Patterns (2024 Competition Data)

| Competition Type | Avg Shake-up | Max Position Change | Primary Cause |
|------------------|--------------|---------------------|---------------|
| Tabular (static) | 5-10%        | ±50 positions       | LB overfitting |
| Time Series      | 15-25%       | ±200 positions      | Temporal shift |
| Computer Vision  | 8-12%        | ±80 positions       | Distribution shift |
| NLP              | 6-10%        | ±60 positions       | Domain variance |
| Medical Imaging  | 12-18%       | ±150 positions      | Site/hospital variation |

### Predicting Shake-up Risk

```python
class ShakeupRiskAnalyzer:
    """
    Analyze risk of shake-up in a competition.
    """
    
    def __init__(self, competition_info):
        self.info = competition_info
        
    def calculate_risk_score(self):
        """
        Calculate shake-up risk score (0-100).
        Higher = more risk.
        """
        risk = 0
        
        # Test set size factor
        if self.info['test_size'] < 5000:
            risk += 30
        elif self.info['test_size'] < 20000:
            risk += 15
            
        # Time-based split factor
        if self.info.get('temporal_split', False):
            risk += 25
            
        # Domain diversity factor
        if self.info.get('multi_domain', False):
            risk += 20
            
        # Historical shake-up factor
        if self.info.get('historical_shakeup', 'low') == 'high':
            risk += 15
            
        return min(risk, 100)
    
    def get_recommendation(self, risk_score):
        """Get strategy recommendation based on risk."""
        if risk_score > 70:
            return {
                'trust_cv': True,
                'cv_folds': 10,
                'ensemble_diversity': 'high',
                'submission_strategy': 'conservative'
            }
        elif risk_score > 40:
            return {
                'trust_cv': True,
                'cv_folds': 5,
                'ensemble_diversity': 'medium',
                'submission_strategy': 'balanced'
            }
        else:
            return {
                'trust_cv': False,
                'cv_folds': 5,
                'ensemble_diversity': 'low',
                'submission_strategy': 'lb_optimized'
            }
```

---

## CV Architecture Patterns

### The CV Strategy Decision Tree

```
                    Data Type?
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    Tabular      Time Series      Image/Text
        │              │              │
   Grouped?      Temporal?      Stratified?
        │              │              │
    ┌───┴───┐    ┌───┴───┐    ┌───┴───┐
    │       │    │       │    │       │
   Yes     No   Yes     No   Yes     No
    │       │    │       │    │       │
 GroupKFold KFold TimeSplit KFold Stratified KFold
```

### CV Strategy Selection Flowchart

```python
def get_cv_strategy(data, target_col, group_col=None, time_col=None, 
                    problem_type='classification'):
    """
    Automatically select appropriate CV strategy.
    
    Args:
        data: DataFrame
        target_col: Target column name
        group_col: Group column for grouped CV (optional)
        time_col: Time column for temporal CV (optional)
        problem_type: 'classification' or 'regression'
        
    Returns:
        CV splitter object
    """
    from sklearn.model_selection import (
        KFold, StratifiedKFold, GroupKFold, 
        TimeSeriesSplit
    )
    
    # Check for time-based structure
    if time_col is not None:
        print("Using TimeSeriesSplit for temporal data")
        return TimeSeriesSplit(n_splits=5)
    
    # Check for group structure
    if group_col is not None:
        n_groups = data[group_col].nunique()
        n_splits = min(5, n_groups)
        print(f"Using GroupKFold with {n_splits} splits")
        return GroupKFold(n_splits=n_splits)
    
    # Standard stratified for classification
    if problem_type == 'classification':
        print("Using StratifiedKFold")
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Standard KFold for regression
    print("Using KFold")
    return KFold(n_splits=5, shuffle=True, random_state=42)
```

### CV Architecture Patterns Table

| Pattern | Use Case | Implementation | Pros | Cons |
|---------|----------|----------------|------|------|
| **KFold** | Standard tabular | `KFold(n_splits=5)` | Simple, fast | No stratification |
| **StratifiedKFold** | Imbalanced classification | `StratifiedKFold(n_splits=5)` | Preserves class ratio | Doesn't handle groups |
| **GroupKFold** | Grouped data (same user/item) | `GroupKFold(n_splits=5)` | Prevents leakage | Requires group column |
| **TimeSeriesSplit** | Temporal data | `TimeSeriesSplit(n_splits=5)` | Respects time order | Smaller training sets |
| **StratifiedGroupKFold** | Grouped + imbalanced | Custom implementation | Best of both | More complex |

---

## CV-LB Correlation Validation

### Measuring CV-LB Correlation

```python
import numpy as np
from scipy.stats import spearmanr, pearsonr

class CVLBCorrelationValidator:
    """
    Validate correlation between CV and LB scores.
    """
    
    def __init__(self):
        self.history = []
        
    def add_submission(self, cv_mean, cv_std, lb_score, model_name):
        """Record a submission for correlation analysis."""
        self.history.append({
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'lb_score': lb_score,
            'model_name': model_name
        })
    
    def calculate_correlation(self):
        """
        Calculate correlation between CV and LB.
        
        Returns:
            Dictionary with correlation metrics
        """
        if len(self.history) < 3:
            return {'error': 'Need at least 3 submissions'}
        
        cv_scores = [h['cv_mean'] for h in self.history]
        lb_scores = [h['lb_score'] for h in self.history]
        
        pearson_r, pearson_p = pearsonr(cv_scores, lb_scores)
        spearman_r, spearman_p = spearmanr(cv_scores, lb_scores)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_submissions': len(self.history),
            'reliable': pearson_r > 0.7 and pearson_p < 0.05
        }
    
    def plot_correlation(self):
        """Visualize CV-LB correlation."""
        import matplotlib.pyplot as plt
        
        cv_scores = [h['cv_mean'] for h in self.history]
        lb_scores = [h['lb_score'] for h in self.history]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(cv_scores, lb_scores, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(cv_scores, lb_scores, 1)
        p = np.poly1d(z)
        plt.plot(cv_scores, p(cv_scores), "r--", alpha=0.8)
        
        plt.xlabel('CV Score')
        plt.ylabel('LB Score')
        plt.title('CV-LB Correlation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
```

### Interpreting CV-LB Correlation

| Pearson R | Interpretation | Action |
|-----------|----------------|--------|
| > 0.9     | Excellent CV   | Trust CV completely |
| 0.7 - 0.9 | Good CV        | Trust CV with caution |
| 0.5 - 0.7 | Weak CV        | Need more validation |
| < 0.5     | Poor CV        | CV strategy broken |

---

## Leakage Detection Framework

### Types of Data Leakage

1. **Target Leakage**: Features that incorporate future information
2. **Train-Test Contamination**: Test data influences training
3. **Group Leakage**: Same group in train and test
4. **Temporal Leakage**: Future data in training period

### LeakageDetector Class

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class LeakageDetector:
    """
    Detect potential data leakage in datasets.
    """
    
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.warnings = []
        
    def detect_target_leakage(self, df, target_col):
        """
        Detect features with suspicious correlation to target.
        
        Args:
            df: DataFrame
            target_col: Target column name
            
        Returns:
            List of suspicious features
        """
        suspicious = []
        
        for col in df.columns:
            if col == target_col:
                continue
                
            # Check correlation for numeric features
            if df[col].dtype in ['int64', 'float64']:
                corr = df[col].corr(df[target_col])
                if abs(corr) > self.threshold:
                    suspicious.append({
                        'feature': col,
                        'correlation': corr,
                        'type': 'numeric_high_correlation'
                    })
                    
        return suspicious
    
    def detect_duplicates(self, train_df, test_df):
        """
        Detect duplicate rows between train and test.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Number of duplicates found
        """
        # Find common columns
        common_cols = list(set(train_df.columns) & set(test_df.columns))
        
        # Merge to find duplicates
        merged = pd.merge(
            train_df[common_cols], 
            test_df[common_cols],
            how='inner',
            indicator=True
        )
        
        n_duplicates = len(merged)
        
        if n_duplicates > 0:
            self.warnings.append(
                f"Found {n_duplicates} duplicate rows between train and test!"
            )
            
        return n_duplicates
    
    def detect_group_leakage(self, df, group_col, target_col):
        """
        Detect if target is perfectly predictable from group.
        
        Args:
            df: DataFrame
            group_col: Group identifier column
            target_col: Target column
            """
        # Check if target is constant within groups
        group_target_var = df.groupby(group_col)[target_col].var()
        
        constant_groups = (group_target_var == 0).sum()
        total_groups = df[group_col].nunique()
        
        if constant_groups / total_groups > 0.9:
            self.warnings.append(
                f"Target is constant within {constant_groups}/{total_groups} groups. "
                "Use GroupKFold to prevent leakage."
            )
            
        return constant_groups, total_groups
    
    def detect_temporal_leakage(self, df, time_col, feature_cols):
        """
        Detect features that may incorporate future information.
        
        Args:
            df: DataFrame with time column
            time_col: Time column name
            feature_cols: List of feature columns to check
        """
        df_sorted = df.sort_values(time_col)
        
        suspicious_features = []
        
        for col in feature_cols:
            if df[col].dtype not in ['int64', 'float64']:
                continue
                
            # Check if feature uses future information
            # (e.g., rolling means that extend beyond current row)
            rolling_mean = df_sorted[col].rolling(window=10, min_periods=1).mean()
            
            # If feature matches rolling mean too well, it may use future info
            correlation = df_sorted[col].corr(rolling_mean.shift(-5))
            
            if correlation > 0.99:
                suspicious_features.append(col)
                
        return suspicious_features
    
    def run_full_diagnostic(self, train_df, test_df=None, 
                           target_col=None, group_col=None, time_col=None):
        """
        Run complete leakage diagnostic.
        
        Returns:
            Diagnostic report dictionary
        """
        report = {
            'warnings': [],
            'suspicious_features': [],
            'duplicates': 0,
            'recommendations': []
        }
        
        # Target leakage
        if target_col:
            suspicious = self.detect_target_leakage(train_df, target_col)
            report['suspicious_features'].extend(suspicious)
            
        # Duplicate detection
        if test_df is not None:
            n_dups = self.detect_duplicates(train_df, test_df)
            report['duplicates'] = n_dups
            
        # Group leakage
        if group_col and target_col:
            const, total = self.detect_group_leakage(train_df, group_col, target_col)
            if const / total > 0.9:
                report['recommendations'].append(
                    f"Use GroupKFold with group_col='{group_col}'"
                )
                
        # Temporal leakage
        if time_col:
            feature_cols = [c for c in train_df.columns 
                          if c not in [target_col, time_col, group_col]]
            suspicious_temporal = self.detect_temporal_leakage(
                train_df, time_col, feature_cols
            )
            if suspicious_temporal:
                report['warnings'].append(
                    f"Features may use future info: {suspicious_temporal}"
                )
                
        report['warnings'] = self.warnings
        return report
```

### Common Leakage Patterns Table

| Pattern | Detection Method | Prevention | Competition Impact |
|---------|------------------|------------|-------------------|
| **Target in Features** | Correlation > 0.95 | Remove feature | DQ/disqualification |
| **ID Column Leakage** | ID correlates with target | Drop ID columns | Massive overfitting |
| **Future Information** | Time-based validation fails | TimeSeriesSplit | Private LB crash |
| **Group Overlap** | Same group in train/test | GroupKFold | Moderate overfitting |
| **Preprocessing Leakage** | Fit on full data | Fit only on train | Subtle overfitting |

---

## Advanced CV Strategies

### Stratified Group KFold

```python
import numpy as np
from sklearn.model_selection import BaseCrossValidator

class StratifiedGroupKFold(BaseCrossValidator):
    """
    Stratified Group K-Fold cross-validator.
    
    Provides train/test indices to split data into k-fold 
    while respecting groups and preserving class distribution.
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X, y, groups):
        """Generate indices for stratified group k-fold."""
        if groups is None:
            raise ValueError("groups must be provided")
            
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if self.n_splits > n_groups:
            raise ValueError(
                f"n_splits ({self.n_splits}) cannot be greater than "
                f"number of groups ({n_groups})"
            )
        
        # Calculate class distribution per group
        group_class_dist = {}
        for group in unique_groups:
            mask = groups == group
            group_y = y[mask]
            # Store majority class for stratification
            group_class_dist[group] = np.bincount(group_y).argmax()
        
        # Sort groups by class for stratification
        sorted_groups = sorted(
            unique_groups, 
            key=lambda g: group_class_dist[g]
        )
        
        # Assign folds
        folds = [[] for _ in range(self.n_splits)]
        for i, group in enumerate(sorted_groups):
            folds[i % self.n_splits].append(group)
        
        # Generate splits
        for i in range(self.n_splits):
            test_groups = folds[i]
            train_groups = [
                g for j, f in enumerate(folds) 
                for g in f if j != i
            ]
            
            test_mask = np.isin(groups, test_groups)
            train_mask = ~test_mask
            
            yield np.where(train_mask)[0], np.where(test_mask)[0]
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
```

### Purged Cross-Validation for Time Series

```python
class PurgedGroupTimeSeriesSplit:
    """
    Time series CV with purge and embargo periods.
    
    Prevents leakage by removing observations close to test period.
    Based on Lopez de Prado's Advances in Financial Machine Learning.
    """
    
    def __init__(self, n_splits=5, purge_gap=0, embargo_pct=0.01):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        
    def split(self, X, y, groups=None):
        """Generate purged time series splits."""
        n_samples = len(X)
        
        # Calculate fold boundaries
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Define test set
            test_start = (i + 1) * fold_size
            test_end = (i + 2) * fold_size if i < self.n_splits - 1 else n_samples
            
            # Apply embargo to test set
            embargo_size = int((test_end - test_start) * self.embargo_pct)
            test_end -= embargo_size
            
            # Define train set with purge gap
            train_end = test_start - self.purge_gap
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
```

---

## CV Best Practices Checklist

- [ ] Analyze data structure before choosing CV strategy
- [ ] Check for groups, time, and class imbalance
- [ ] Run leakage detection before any modeling
- [ ] Validate CV-LB correlation with initial submissions
- [ ] Use at least 5 folds for reliable estimates
- [ ] Set random_state for reproducibility
- [ ] Monitor CV standard deviation (should be < 0.01 for stable data)
- [ ] Document CV strategy and rationale
- [ ] Test ensemble on full CV before LB submission
- [ ] Save OOF predictions for stacking

---

## Key Takeaways

1. **CV is your true leaderboard**—optimize for robust CV, not public LB
2. **Understand your data structure**—groups, time, and imbalance drive CV choice
3. **Detect leakage early**—it's easier to prevent than fix
4. **Validate CV-LB correlation**—at least 3 submissions to establish trust
5. **Use appropriate CV for your data**—one size does not fit all
