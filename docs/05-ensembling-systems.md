# Part 5: Ensembling & Stacking Systems

## The Power of Ensembles

Ensembling is the technique that separates good competitors from medalists. While a single model might achieve 0.80 AUC, a well-constructed ensemble can push this to 0.82-0.85—a massive difference in competitive rankings.

### Why Ensembles Work

```
Model Diversity → Error Decorrelation → Variance Reduction → Better Performance
```

Individual models make different errors. When averaged, random errors cancel while systematic patterns reinforce.

### The Bias-Variance Tradeoff in Ensembles

```
Single Model:     High Variance, Low Bias (if well-tuned)
Small Ensemble:   Medium Variance, Low Bias
Large Ensemble:   Low Variance, Low Bias (diminishing returns)
```

---

## Dimensions of Diversity

Effective ensembles require diversity across multiple dimensions:

### Diversity Dimensions Table

| Dimension | Examples | Impact | Implementation |
|-----------|----------|--------|----------------|
| **Algorithm** | LightGBM, XGBoost, CatBoost, Neural Net | High | Train different model types |
| **Features** | Different feature subsets | High | Feature bagging, random selection |
| **Data Samples** | Different bootstrap samples | Medium | Bagging, pasting |
| **Hyperparameters** | Different learning rates, depths | Medium | Grid of configurations |
| **Initialization** | Different random seeds | Low | Multiple seeds per config |
| **CV Folds** | Models trained on different folds | Medium | Fold-specific models |

### Diversity Measurement

```python
import numpy as np
from scipy.stats import spearmanr

def measure_diversity(predictions_dict, metric='correlation'):
    """
    Measure diversity between model predictions.
    
    Args:
        predictions_dict: Dict of {model_name: predictions_array}
        metric: 'correlation', 'covariance', or 'disagreement'
        
    Returns:
        Diversity matrix and average diversity score
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Create prediction matrix
    pred_matrix = np.column_stack([
        predictions_dict[name] for name in model_names
    ])
    
    if metric == 'correlation':
        # Average correlation (lower = more diverse)
        corr_matrix = np.corrcoef(pred_matrix.T)
        # Exclude diagonal
        mask = ~np.eye(n_models, dtype=bool)
        avg_diversity = 1 - np.mean(corr_matrix[mask])
        
    elif metric == 'covariance':
        # Covariance-based diversity
        cov_matrix = np.cov(pred_matrix.T)
        mask = ~np.eye(n_models, dtype=bool)
        avg_diversity = 1 - np.mean(cov_matrix[mask])
        
    elif metric == 'disagreement':
        # Binary disagreement for classification
        binary_preds = (pred_matrix > 0.5).astype(int)
        n_samples = len(binary_preds)
        
        disagreements = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                disagreement = np.mean(binary_preds[:, i] != binary_preds[:, j])
                disagreements.append(disagreement)
        
        avg_diversity = np.mean(disagreements)
        corr_matrix = None
    
    return {
        'diversity_matrix': corr_matrix,
        'avg_diversity': avg_diversity,
        'model_names': model_names
    }


def select_diverse_models(predictions_dict, n_select=5, threshold=0.95):
    """
    Select most diverse subset of models.
    
    Args:
        predictions_dict: Dict of model predictions
        n_select: Number of models to select
        threshold: Maximum allowed correlation
        
    Returns:
        List of selected model names
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Compute correlation matrix
    pred_matrix = np.column_stack([
        predictions_dict[name] for name in model_names
    ])
    corr_matrix = np.corrcoef(pred_matrix.T)
    
    # Greedy selection: start with best model, add most diverse
    selected = [model_names[0]]
    selected_idx = [0]
    
    while len(selected) < n_select:
        best_diversity = -1
        best_model = None
        best_idx = None
        
        for i, name in enumerate(model_names):
            if i in selected_idx:
                continue
            
            # Check correlation with all selected models
            max_corr = max(corr_matrix[i, j] for j in selected_idx)
            
            if max_corr < threshold:
                # Calculate average diversity
                avg_corr = np.mean([corr_matrix[i, j] for j in selected_idx])
                diversity = 1 - avg_corr
                
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_model = name
                    best_idx = i
        
        if best_model is None:
            break
            
        selected.append(best_model)
        selected_idx.append(best_idx)
    
    return selected
```

---

## Stacking Architecture

### Two-Level Stacking Diagram

```
Level 0 (Base Models):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ LightGBM │  │ XGBoost  │  │ CatBoost │  │  Neural  │
│  (OOF)   │  │  (OOF)   │  │  (OOF)   │  │  (OOF)   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Meta-Features       │
              │ (Stacked Predictions) │
              └───────────┬───────────┘
                          │
Level 1 (Meta-Model):     ▼
              ┌───────────────────────┐
              │    Meta-Learner       │
              │   (LightGBM/Ridge)    │
              └───────────────────────┘
```

### Complete Stacking Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
import lightgbm as lgb

class StackingEnsemble:
    """
    Complete stacking ensemble implementation.
    
    Supports multiple base models and configurable meta-learners.
    """
    
    def __init__(self, base_models, meta_model=None, 
                 n_folds=5, use_proba=True, random_state=42):
        """
        Args:
            base_models: Dict of {name: model_instance}
            meta_model: Meta-learner (default: LightGBM)
            n_folds: Number of CV folds for OOF predictions
            use_proba: Use probabilities for classification
            random_state: Random seed
        """
        self.base_models = base_models
        self.meta_model = meta_model or lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=random_state
        )
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.random_state = random_state
        
        self.trained_base_models = {}
        self.oof_predictions = None
        self.meta_features_names = []
        
    def fit(self, X, y, groups=None):
        """
        Fit stacking ensemble.
        
        Args:
            X: Features
            y: Target
            groups: Group labels for grouped CV
        """
        # Determine CV strategy
        if groups is not None:
            from sklearn.model_selection import GroupKFold
            kf = GroupKFold(n_splits=self.n_folds)
            split_args = {'groups': groups}
        elif y.nunique() == 2:
            kf = StratifiedKFold(
                n_splits=self.n_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
            split_args = {}
        else:
            kf = KFold(
                n_splits=self.n_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
            split_args = {}
        
        n_samples = len(X)
        n_models = len(self.base_models)
        
        # Initialize OOF predictions array
        if self.use_proba and y.nunique() == 2:
            self.oof_predictions = np.zeros((n_samples, n_models))
        else:
            self.oof_predictions = np.zeros((n_samples, n_models))
        
        # Store trained models per fold
        self.trained_base_models = {name: [] for name in self.base_models}
        
        # Generate OOF predictions for each base model
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            print(f"Training base model: {name}")
            
            oof_preds = np.zeros(n_samples)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, **split_args)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone and train model
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                # Store trained model
                self.trained_base_models[name].append(model_clone)
                
                # Generate OOF predictions
                if self.use_proba and hasattr(model_clone, 'predict_proba'):
                    preds = model_clone.predict_proba(X_val)[:, 1]
                else:
                    preds = model_clone.predict(X_val)
                
                oof_preds[val_idx] = preds
            
            self.oof_predictions[:, model_idx] = oof_preds
            self.meta_features_names.append(name)
        
        # Train meta-learner on OOF predictions
        print("Training meta-learner...")
        self.meta_model.fit(self.oof_predictions, y)
        
        return self
    
    def predict(self, X):
        """
        Predict using trained stacking ensemble.
        
        Args:
            X: Features
            
        Returns:
            Final predictions
        """
        # Generate meta-features from base models
        meta_features = self._generate_meta_features(X)
        
        # Meta-learner prediction
        if self.use_proba and hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)[:, 1]
        else:
            return self.meta_model.predict(meta_features)
    
    def _generate_meta_features(self, X):
        """Generate meta-features by averaging base model predictions."""
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        for model_idx, (name, _) in enumerate(self.base_models.items()):
            fold_preds = []
            
            for fold_model in self.trained_base_models[name]:
                if self.use_proba and hasattr(fold_model, 'predict_proba'):
                    preds = fold_model.predict_proba(X)[:, 1]
                else:
                    preds = fold_model.predict(X)
                fold_preds.append(preds)
            
            # Average across folds
            meta_features[:, model_idx] = np.mean(fold_preds, axis=0)
        
        return meta_features
    
    def get_feature_importance(self):
        """Get importance of each base model."""
        if hasattr(self.meta_model, 'feature_importances_'):
            importance = self.meta_model.feature_importances_
            return dict(zip(self.meta_features_names, importance))
        return None


# Usage example
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = {
    'lgb': lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05),
    'xgb': xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05),
    'cat': cb.CatBoostClassifier(iterations=1000, learning_rate=0.05, verbose=False),
    'rf': RandomForestClassifier(n_estimators=200, n_jobs=-1),
}

# Create and train stacking ensemble
stacker = StackingEnsemble(
    base_models=base_models,
    meta_model=lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05),
    n_folds=5
)

stacker.fit(X_train, y_train)
predictions = stacker.predict(X_test)
"""
```

---

## Hill Climbing for Weight Optimization

### Hill Climbing Algorithm

```python
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

class HillClimbingOptimizer:
    """
    Hill climbing optimization for ensemble weights.
    
    Finds optimal weights by iteratively improving from a starting point.
    """
    
    def __init__(self, metric='auc', maximize=True, precision=4):
        """
        Args:
            metric: 'auc', 'rmse', 'logloss'
            maximize: Whether to maximize (True) or minimize (False)
            precision: Decimal precision for weights
        """
        self.metric = metric
        self.maximize = maximize
        self.precision = precision
        self.weights = None
        self.best_score = None
        
    def optimize(self, predictions_dict, y_true, 
                 initial_weights=None, max_iter=1000):
        """
        Optimize ensemble weights.
        
        Args:
            predictions_dict: Dict of {model_name: predictions}
            y_true: True labels
            initial_weights: Starting weights (default: equal)
            max_iter: Maximum iterations
            
        Returns:
            Optimal weights dict
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        # Initialize weights
        if initial_weights is None:
            weights = np.ones(n_models) / n_models
        else:
            weights = np.array([initial_weights.get(m, 1/n_models) 
                               for m in model_names])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Get prediction matrix
        pred_matrix = np.column_stack([
            predictions_dict[name] for name in model_names
        ])
        
        # Evaluate initial score
        best_score = self._evaluate(pred_matrix, weights, y_true)
        best_weights = weights.copy()
        
        print(f"Initial score: {best_score:.6f}")
        
        # Hill climbing
        step_size = 0.01
        no_improve_count = 0
        
        for iteration in range(max_iter):
            improved = False
            
            # Try adjusting each weight
            for i in range(n_models):
                for direction in [-1, 1]:
                    # Create new weights
                    new_weights = best_weights.copy()
                    new_weights[i] += direction * step_size
                    
                    # Ensure non-negative and normalize
                    new_weights = np.maximum(new_weights, 0)
                    new_weights = new_weights / new_weights.sum()
                    
                    # Evaluate
                    score = self._evaluate(pred_matrix, new_weights, y_true)
                    
                    # Check if better
                    if self._is_better(score, best_score):
                        best_score = score
                        best_weights = new_weights
                        improved = True
                        no_improve_count = 0
            
            if not improved:
                no_improve_count += 1
                
                # Reduce step size if no improvement
                if no_improve_count >= 10:
                    step_size /= 2
                    no_improve_count = 0
                    
                    if step_size < 0.0001:
                        print(f"Converged at iteration {iteration}")
                        break
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: score = {best_score:.6f}")
        
        # Round weights to precision
        self.weights = {
            name: round(w, self.precision) 
            for name, w in zip(model_names, best_weights)
        }
        self.best_score = best_score
        
        return self.weights
    
    def _evaluate(self, pred_matrix, weights, y_true):
        """Evaluate ensemble with given weights."""
        ensemble_pred = np.average(pred_matrix, axis=1, weights=weights)
        
        if self.metric == 'auc':
            return roc_auc_score(y_true, ensemble_pred)
        elif self.metric == 'rmse':
            return -mean_squared_error(y_true, ensemble_pred, squared=False)
        elif self.metric == 'logloss':
            from sklearn.metrics import log_loss
            # Clip for numerical stability
            ensemble_pred = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
            return -log_loss(y_true, ensemble_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _is_better(self, score, best_score):
        """Check if score is better than best."""
        if self.maximize:
            return score > best_score
        else:
            return score < best_score
    
    def predict(self, predictions_dict):
        """Make prediction with optimized weights."""
        if self.weights is None:
            raise ValueError("Must call optimize() first")
        
        model_names = list(predictions_dict.keys())
        pred_matrix = np.column_stack([
            predictions_dict[name] for name in model_names
        ])
        weights = np.array([self.weights.get(m, 0) for m in model_names])
        
        return np.average(pred_matrix, axis=1, weights=weights)


# Usage example
"""
# Get OOF predictions from multiple models
oof_preds = {
    'lgb': lgb_oof_preds,
    'xgb': xgb_oof_preds,
    'cat': cat_oof_preds,
    'nn': nn_oof_preds,
}

# Optimize weights
optimizer = HillClimbingOptimizer(metric='auc', maximize=True)
weights = optimizer.optimize(oof_preds, y_train)

print("Optimized weights:", weights)
print("Best CV score:", optimizer.best_score)

# Predict on test set
test_preds = {
    'lgb': lgb_test_preds,
    'xgb': xgb_test_preds,
    'cat': cat_test_preds,
    'nn': nn_test_preds,
}
final_predictions = optimizer.predict(test_preds)
"""
```

---

## Ensemble Failure Modes

### Common Ensemble Pitfalls

| Failure Mode | Cause | Detection | Solution |
|--------------|-------|-----------|----------|
| **Overfitting** | Too many correlated models | CV << LB | Remove correlated models, regularize |
| **Under-diversity** | Models too similar | High correlation (>0.95) | Add diverse model types |
| **Weight Instability** | Small sample size | Weights change with new folds | Constrain weights, use regularization |
| **Meta-learner Overfit** | Too complex meta-model | Meta CV << base CV | Simpler meta-learner, more folds |
| **Leakage** | Using test data in ensemble | Impossible CV scores | Strict train/validation separation |

### Ensemble Size Tradeoff Analysis

```python
def analyze_ensemble_size(predictions_list, y_true, max_size=20):
    """
    Analyze optimal ensemble size.
    
    Args:
        predictions_list: List of (name, predictions) tuples
        y_true: True labels
        max_size: Maximum ensemble size to test
        
    Returns:
        Analysis results
    """
    results = []
    
    for size in range(1, min(max_size + 1, len(predictions_list) + 1)):
        # Take top 'size' models
        subset = predictions_list[:size]
        
        # Simple average ensemble
        ensemble_pred = np.mean([p for _, p in subset], axis=0)
        score = roc_auc_score(y_true, ensemble_pred)
        
        results.append({
            'ensemble_size': size,
            'cv_score': score,
            'models': [name for name, _ in subset]
        })
    
    return pd.DataFrame(results)


# Plot ensemble size vs performance
"""
import matplotlib.pyplot as plt

results = analyze_ensemble_size(predictions_list, y_true)

plt.figure(figsize=(10, 6))
plt.plot(results['ensemble_size'], results['cv_score'], marker='o')
plt.xlabel('Ensemble Size')
plt.ylabel('CV Score')
plt.title('Ensemble Size vs Performance')
plt.grid(True, alpha=0.3)
plt.axhline(y=results['cv_score'].max(), color='r', linestyle='--', 
            label=f'Best: {results["cv_score"].max():.4f}')
plt.legend()
plt.show()
"""
```

---

## Tradeoff Analysis: Competition vs Production

### Ensemble Size Guidelines

| Context | Max Ensemble Size | Latency Budget | Optimization |
|---------|-------------------|----------------|--------------|
| Kaggle Competition | 50+ models | N/A | Pure accuracy |
| Research/Experiment | 20-30 models | Flexible | Accuracy + diversity |
| Batch Scoring | 10-20 models | Minutes | Accuracy + throughput |
| Real-time API | 3-5 models | <100ms | Latency-constrained |
| Edge/Mobile | 1-2 models | <10ms | Model compression |

### Production Ensemble Strategy

```python
class ProductionEnsemble:
    """
    Production-optimized ensemble.
    
    Balances accuracy with latency constraints.
    """
    
    def __init__(self, latency_budget_ms=50):
        self.latency_budget_ms = latency_budget_ms
        self.models = []
        self.weights = []
        self.model_latencies = {}
        
    def add_model(self, model, weight, latency_ms):
        """Add model to ensemble with latency tracking."""
        self.models.append(model)
        self.weights.append(weight)
        self.model_latencies[len(self.models) - 1] = latency_ms
        
    def can_add_model(self, latency_ms):
        """Check if adding model stays within budget."""
        total_latency = sum(self.model_latencies.values()) + latency_ms
        
        # Assume parallel execution (max latency)
        # For sequential, use sum
        max_latency = max(
            list(self.model_latencies.values()) + [latency_ms]
        )
        
        return max_latency <= self.latency_budget_ms
    
    def predict(self, X):
        """Make prediction with all models."""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(self.weights)
        weights = weights / weights.sum()
        
        return np.average(predictions, axis=0, weights=weights)
```

---

## Ensembling Best Practices Checklist

- [ ] Ensure model diversity (algorithm, features, hyperparameters)
- [ ] Measure diversity before building ensemble
- [ ] Use OOF predictions for stacking
- [ ] Start with simple averaging before complex stacking
- [ ] Optimize weights with hill climbing or CV
- [ ] Validate ensemble doesn't overfit
- [ ] Test ensemble on holdout set
- [ ] Consider latency constraints for production
- [ ] Document ensemble composition and rationale
- [ ] Save all base model predictions for reproducibility

---

## Key Takeaways

1. **Diversity is key**—correlated models don't help
2. **Start simple**—weighted average before stacking
3. **Use OOF predictions**—never leak validation data
4. **Optimize weights**—hill climbing finds better combinations
5. **Monitor for overfitting**—more models ≠ better performance
6. **Production needs constraints**—latency limits ensemble size
7. **Document everything**—ensemble composition, weights, rationale
