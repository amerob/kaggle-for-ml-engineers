# Part 6: Advanced Tactics

## Beyond Standard Techniques

This chapter covers advanced techniques that can provide the final percentage points needed for top competition placements. These tactics require careful implementation to avoid overfitting and leakage.

---

## Pseudo-Labeling

### Concept

Pseudo-labeling uses a model's confident predictions on unlabeled test data as additional training examples. This effectively increases training data size and can improve model performance.

```
Train Model → Predict on Test → Select Confident Predictions 
    → Add to Training → Retrain Model
```

### PseudoLabeler Implementation

```python
import numpy as np
import pandas as pd
from sklearn.base import clone

class PseudoLabeler:
    """
    Pseudo-labeling with confidence thresholding.
    
    Adds high-confidence test predictions to training data
    for iterative model improvement.
    """
    
    def __init__(self, base_model, threshold=0.9, max_iterations=3,
                 sample_ratio=0.5, verbose=True):
        """
        Args:
            base_model: Base model to train
            threshold: Confidence threshold for pseudo-labels
            max_iterations: Maximum pseudo-labeling iterations
            sample_ratio: Ratio of test samples to use
            verbose: Print progress
        """
        self.base_model = base_model
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.sample_ratio = sample_ratio
        self.verbose = verbose
        
        self.models = []
        self.pseudo_labels_history = []
        
    def fit(self, X_train, y_train, X_test):
        """
        Fit with pseudo-labeling.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (unlabeled)
            
        Returns:
            Final trained model
        """
        current_X = X_train.copy()
        current_y = y_train.copy()
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n=== Pseudo-Labeling Iteration {iteration + 1} ===")
                print(f"Training data size: {len(current_X)}")
            
            # Train model on current data
            model = clone(self.base_model)
            model.fit(current_X, current_y)
            self.models.append(model)
            
            # Predict on test set
            if hasattr(model, 'predict_proba'):
                test_probs = model.predict_proba(X_test)
                
                # For binary classification
                if test_probs.shape[1] == 2:
                    confidence = np.maximum(test_probs[:, 0], test_probs[:, 1])
                    predictions = test_probs[:, 1]
                else:
                    # Multi-class
                    confidence = np.max(test_probs, axis=1)
                    predictions = np.argmax(test_probs, axis=1)
            else:
                predictions = model.predict(X_test)
                confidence = np.ones(len(predictions))  # No confidence info
            
            # Select high-confidence samples
            confident_mask = confidence >= self.threshold
            n_confident = confident_mask.sum()
            
            if self.verbose:
                print(f"Confident predictions: {n_confident} / {len(X_test)}")
            
            if n_confident == 0:
                if self.verbose:
                    print("No confident predictions, stopping.")
                break
            
            # Limit number of pseudo-labels
            max_pseudo = int(len(X_test) * self.sample_ratio)
            
            if n_confident > max_pseudo:
                # Select top confident samples
                confident_indices = np.where(confident_mask)[0]
                top_indices = confident_indices[
                    np.argsort(confidence[confident_indices])[-max_pseudo:]
                ]
                confident_mask = np.zeros(len(X_test), dtype=bool)
                confident_mask[top_indices] = True
                n_confident = max_pseudo
                
                if self.verbose:
                    print(f"Limited to top {n_confident} pseudo-labels")
            
            # Add pseudo-labeled data to training
            pseudo_X = X_test[confident_mask]
            pseudo_y = predictions[confident_mask]
            
            self.pseudo_labels_history.append({
                'iteration': iteration + 1,
                'n_pseudo_labels': n_confident,
                'mean_confidence': confidence[confident_mask].mean(),
                'mean_prediction': pseudo_y.mean() if len(pseudo_y) > 0 else 0
            })
            
            # Combine data
            current_X = pd.concat([X_train, pseudo_X], ignore_index=True)
            current_y = pd.concat([y_train, pd.Series(pseudo_y)], ignore_index=True)
        
        return self.models[-1] if self.models else None
    
    def predict(self, X):
        """Predict using final model."""
        if not self.models:
            raise ValueError("Model not fitted yet")
        
        final_model = self.models[-1]
        
        if hasattr(final_model, 'predict_proba'):
            return final_model.predict_proba(X)[:, 1]
        else:
            return final_model.predict(X)
    
    def get_history(self):
        """Get pseudo-labeling history."""
        return pd.DataFrame(self.pseudo_labels_history)


# Safety-conscious pseudo-labeling
class SafePseudoLabeler(PseudoLabeler):
    """
    Pseudo-labeler with additional safety checks.
    
    Prevents common pseudo-labeling pitfalls.
    """
    
    def __init__(self, base_model, threshold=0.9, max_iterations=3,
                 sample_ratio=0.5, verbose=True,
                 max_class_imbalance=3.0,  # Max ratio between classes
                 min_cv_improvement=0.001):  # Min CV improvement to continue
        super().__init__(base_model, threshold, max_iterations, 
                        sample_ratio, verbose)
        self.max_class_imbalance = max_class_imbalance
        self.min_cv_improvement = min_cv_improvement
        self.cv_scores = []
        
    def fit(self, X_train, y_train, X_test, X_val=None, y_val=None):
        """
        Fit with safety checks.
        
        Args:
            X_val, y_val: Validation set for monitoring
        """
        current_X = X_train.copy()
        current_y = y_train.copy()
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n=== Safe Pseudo-Labeling Iteration {iteration + 1} ===")
            
            # Train model
            model = clone(self.base_model)
            model.fit(current_X, current_y)
            
            # Validate if validation set provided
            if X_val is not None and y_val is not None:
                if hasattr(model, 'predict_proba'):
                    val_pred = model.predict_proba(X_val)[:, 1]
                    from sklearn.metrics import roc_auc_score
                    val_score = roc_auc_score(y_val, val_pred)
                else:
                    val_pred = model.predict(X_val)
                    from sklearn.metrics import accuracy_score
                    val_score = accuracy_score(y_val, val_pred)
                
                self.cv_scores.append(val_score)
                
                if self.verbose:
                    print(f"Validation score: {val_score:.4f}")
                
                # Check for improvement
                if iteration > 0:
                    improvement = val_score - self.cv_scores[-2]
                    if improvement < self.min_cv_improvement:
                        if self.verbose:
                            print(f"Insufficient improvement ({improvement:.4f}), stopping.")
                        break
            
            # Get pseudo-labels
            test_probs = model.predict_proba(X_test)[:, 1]
            confidence = np.maximum(test_probs, 1 - test_probs)
            
            # Select confident predictions
            confident_mask = confidence >= self.threshold
            
            # Check class balance
            pseudo_labels = (test_probs[confident_mask] > 0.5).astype(int)
            if len(pseudo_labels) > 0:
                class_counts = np.bincount(pseudo_labels)
                if len(class_counts) == 2:
                    imbalance = class_counts.max() / (class_counts.min() + 1)
                    if imbalance > self.max_class_imbalance:
                        if self.verbose:
                            print(f"Class imbalance too high ({imbalance:.1f}), adjusting...")
                        # Reduce threshold to get more balanced labels
                        confident_mask = confidence >= (self.threshold * 0.8)
            
            # Add to training data
            pseudo_X = X_test[confident_mask]
            pseudo_y = (test_probs[confident_mask] > 0.5).astype(int)
            
            current_X = pd.concat([current_X, pseudo_X], ignore_index=True)
            current_y = pd.concat([current_y, pd.Series(pseudo_y)], ignore_index=True)
            
            self.models.append(model)
        
        return self.models[-1] if self.models else None
```

### Pseudo-Labeling Best Practices

| Consideration | Recommendation | Rationale |
|---------------|----------------|-----------|
| **Threshold** | 0.9-0.95 for binary | Higher = more confident, fewer labels |
| **Max Iterations** | 2-3 | Diminishing returns, overfitting risk |
| **Sample Ratio** | 0.3-0.5 of test set | Prevents test set dominance |
| **Validation** | Monitor CV score | Stop if no improvement |
| **Class Balance** | Enforce balance | Prevents biased pseudo-labels |

---

## Curriculum Training

### Concept

Curriculum training presents training examples in order of increasing difficulty, helping models learn better representations.

### CurriculumTrainer Implementation

```python
import numpy as np
from sklearn.base import clone

class CurriculumTrainer:
    """
    Curriculum learning trainer.
    
    Trains model on progressively harder examples.
    """
    
    def __init__(self, base_model, difficulty_metric='prediction_confidence',
                 n_stages=3, stage_ratio=0.5):
        """
        Args:
            base_model: Model to train
            difficulty_metric: How to measure example difficulty
            n_stages: Number of curriculum stages
            stage_ratio: Ratio of data to add each stage
        """
        self.base_model = base_model
        self.difficulty_metric = difficulty_metric
        self.n_stages = n_stages
        self.stage_ratio = stage_ratio
        self.models = []
        
    def calculate_difficulty(self, X, y, model=None):
        """
        Calculate difficulty score for each example.
        
        Lower score = easier example
        """
        if model is None:
            # Initial difficulty: use feature-based heuristic
            # Examples closer to class centroid are easier
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate distance to class mean
            class_0_mean = X_scaled[y == 0].mean(axis=0)
            class_1_mean = X_scaled[y == 1].mean(axis=0)
            
            distances = []
            for i, (x, label) in enumerate(zip(X_scaled, y)):
                if label == 0:
                    dist = np.linalg.norm(x - class_0_mean)
                else:
                    dist = np.linalg.norm(x - class_1_mean)
                distances.append(dist)
            
            return np.array(distances)
        
        else:
            # Model-based difficulty
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                
                if self.difficulty_metric == 'prediction_confidence':
                    # Lower confidence = harder example
                    confidence = np.max(probs, axis=1)
                    return 1 - confidence
                
                elif self.difficulty_metric == 'prediction_error':
                    # Higher error = harder example
                    pred_classes = np.argmax(probs, axis=1)
                    errors = (pred_classes != y).astype(float)
                    return errors
            
            else:
                # No probability, use prediction disagreement
                preds = model.predict(X)
                errors = (preds != y).astype(float)
                return errors
    
    def fit(self, X, y):
        """
        Fit with curriculum learning.
        
        Args:
            X: Training features
            y: Training labels
        """
        n_samples = len(X)
        
        # Calculate initial difficulties
        difficulties = self.calculate_difficulty(X, y)
        
        # Sort by difficulty
        sorted_indices = np.argsort(difficulties)
        
        # Determine stage sizes
        stage_sizes = []
        current_size = int(n_samples * self.stage_ratio)
        for _ in range(self.n_stages):
            stage_sizes.append(min(current_size, n_samples))
            current_size = int(current_size / self.stage_ratio)
        
        # Train in stages
        for stage, stage_size in enumerate(stage_sizes):
            print(f"\n=== Curriculum Stage {stage + 1}/{self.n_stages} ===")
            print(f"Training on {stage_size} easiest examples")
            
            # Select easiest examples
            stage_indices = sorted_indices[:stage_size]
            X_stage = X.iloc[stage_indices] if hasattr(X, 'iloc') else X[stage_indices]
            y_stage = y.iloc[stage_indices] if hasattr(y, 'iloc') else y[stage_indices]
            
            # Train model
            model = clone(self.base_model)
            model.fit(X_stage, y_stage)
            self.models.append(model)
            
            # Recalculate difficulties for next stage
            if stage < self.n_stages - 1:
                difficulties = self.calculate_difficulty(X, y, model)
                sorted_indices = np.argsort(difficulties)
        
        return self.models[-1]
    
    def predict(self, X):
        """Predict using final model."""
        if not self.models:
            raise ValueError("No models trained")
        
        final_model = self.models[-1]
        
        if hasattr(final_model, 'predict_proba'):
            return final_model.predict_proba(X)[:, 1]
        else:
            return final_model.predict(X)


# Advanced: Self-paced learning
class SelfPacedTrainer(CurriculumTrainer):
    """
    Self-paced curriculum learning.
    
    Automatically adjusts difficulty threshold based on model performance.
    """
    
    def __init__(self, base_model, initial_threshold=0.1, 
                 threshold_growth=1.2, target_accuracy=0.9):
        super().__init__(base_model, n_stages=5)
        self.initial_threshold = initial_threshold
        self.threshold_growth = threshold_growth
        self.target_accuracy = target_accuracy
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Fit with self-paced curriculum."""
        threshold = self.initial_threshold
        current_X = X.copy()
        current_y = y.copy()
        
        for stage in range(self.n_stages):
            print(f"\n=== Self-Paced Stage {stage + 1} ===")
            print(f"Current threshold: {threshold:.3f}")
            print(f"Training set size: {len(current_X)}")
            
            # Train model
            model = clone(self.base_model)
            model.fit(current_X, current_y)
            self.models.append(model)
            
            # Evaluate
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                accuracy = (val_pred == y_val).mean()
                print(f"Validation accuracy: {accuracy:.3f}")
                
                if accuracy >= self.target_accuracy:
                    print("Target accuracy reached!")
                    break
            
            # Increase threshold for next stage
            threshold = min(threshold * self.threshold_growth, 1.0)
            
            # Select harder examples
            difficulties = self.calculate_difficulty(X, y, model)
            easy_mask = difficulties <= threshold
            
            current_X = X[easy_mask]
            current_y = y[easy_mask]
        
        return self.models[-1]
```

---

## Knowledge Distillation

### Concept

Knowledge distillation transfers knowledge from a large, complex "teacher" model to a smaller, faster "student" model.

```
Teacher Model (Large, Slow) → Soft Labels → Student Model (Small, Fast)
```

### ModelDistiller Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""
    
    def __init__(self, X, y_teacher, y_true=None):
        self.X = torch.FloatTensor(X)
        self.y_teacher = torch.FloatTensor(y_teacher)
        self.y_true = torch.FloatTensor(y_true) if y_true is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y_true is not None:
            return self.X[idx], self.y_teacher[idx], self.y_true[idx]
        return self.X[idx], self.y_teacher[idx]


class ModelDistiller:
    """
    Knowledge distillation from teacher to student model.
    
    Supports temperature scaling and loss weighting.
    """
    
    def __init__(self, teacher_model, student_model, 
                 temperature=3.0, alpha=0.5, device='cuda'):
        """
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss (1-alpha for hard labels)
            device: 'cuda' or 'cpu'
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        self.teacher_model.to(device)
        self.student_model.to(device)
        self.teacher_model.eval()
        
    def get_teacher_predictions(self, X, batch_size=256):
        """
        Get soft predictions from teacher model.
        
        Args:
            X: Input features
            batch_size: Batch size for inference
            
        Returns:
            Teacher predictions (probabilities)
        """
        self.teacher_model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                outputs = self.teacher_model(batch)
                probs = F.softmax(outputs / self.temperature, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def distillation_loss(self, student_logits, teacher_probs, true_labels=None):
        """
        Compute distillation loss.
        
        Combines:
        1. KL divergence between student and teacher soft predictions
        2. Cross-entropy with true labels (if available)
        """
        # Soft predictions from student
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        kl_loss = F.kl_div(
            student_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        if true_labels is not None and self.alpha < 1.0:
            # Hard label loss
            ce_loss = F.cross_entropy(student_logits, true_labels)
            return self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return kl_loss
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=64, lr=0.001):
        """
        Train student model with distillation.
        
        Args:
            X_train: Training features
            y_train: Training labels (for hard loss)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        # Get teacher predictions
        print("Generating teacher predictions...")
        teacher_preds = self.get_teacher_predictions(X_train)
        
        # Create dataset
        train_dataset = DistillationDataset(X_train, teacher_preds, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.student_model.train()
            total_loss = 0
            
            for batch in train_loader:
                if len(batch) == 3:
                    X_batch, teacher_batch, y_batch = batch
                    y_batch = y_batch.to(self.device)
                else:
                    X_batch, teacher_batch = batch
                    y_batch = None
                
                X_batch = X_batch.to(self.device)
                teacher_batch = teacher_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                student_logits = self.student_model(X_batch)
                
                # Compute loss
                loss = self.distillation_loss(
                    student_logits, teacher_batch, y_batch
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}")
            else:
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        
        return self.student_model
    
    def evaluate(self, X, y):
        """Evaluate student model."""
        self.student_model.eval()
        
        dataset = DistillationDataset(X, np.zeros((len(X), 2)), y)
        loader = DataLoader(dataset, batch_size=256)
        
        total_loss = 0
        with torch.no_grad():
            for X_batch, _, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.student_model(X_batch)
                loss = F.cross_entropy(logits, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def predict(self, X):
        """Make predictions with student model."""
        self.student_model.eval()
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.FloatTensor(X[i:i+256]).to(self.device)
                outputs = self.student_model(batch)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.concatenate(predictions)


# Temperature scaling analysis
def analyze_temperature_effect(teacher_probs, true_labels, temperatures=[1, 2, 3, 5, 10]):
    """
    Analyze effect of temperature on distillation.
    
    Args:
        teacher_probs: Teacher model predictions
        true_labels: True labels
        temperatures: List of temperatures to test
        
    Returns:
        Analysis results
    """
    results = []
    
    for T in temperatures:
        # Apply temperature scaling
        scaled_probs = np.power(teacher_probs, 1/T)
        scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
        
        # Calculate entropy (higher = softer labels)
        entropy = -np.sum(scaled_probs * np.log(scaled_probs + 1e-10), axis=1).mean()
        
        # Calculate accuracy
        predictions = np.argmax(scaled_probs, axis=1)
        accuracy = (predictions == true_labels).mean()
        
        results.append({
            'temperature': T,
            'entropy': entropy,
            'accuracy': accuracy
        })
    
    return pd.DataFrame(results)
```

### Distillation Best Practices

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| **Temperature** | 2-5 | Higher = softer labels, more information transfer |
| **Alpha** | 0.3-0.7 | Balance between distillation and hard labels |
| **Student Size** | 10-50% of teacher | Significant compression while maintaining accuracy |
| **Training Data** | Same as teacher | Can use unlabeled data with teacher predictions |

---

## Safety Considerations

### When Advanced Tactics Can Backfire

| Tactic | Risk | Mitigation |
|--------|------|------------|
| **Pseudo-Labeling** | Test set leakage, overfitting | Validate on holdout, monitor CV |
| **Curriculum Learning** | Wrong difficulty measure | Use multiple metrics, validate |
| **Distillation** | Teacher errors propagate | High-quality teacher, temperature tuning |

### Validation Checklist for Advanced Tactics

- [ ] Verify no data leakage in implementation
- [ ] Monitor CV score throughout process
- [ ] Compare to baseline (no advanced tactics)
- [ ] Test on holdout set before submission
- [ ] Document all hyperparameters
- [ ] Ensure reproducibility

---

## Key Takeaways

1. **Pseudo-labeling amplifies training data**—use with confidence thresholds
2. **Curriculum learning helps convergence**—start easy, increase difficulty
3. **Distillation enables model compression**—teacher knowledge → student efficiency
4. **All tactics require validation**—monitor CV, stop if no improvement
5. **Safety first**—one bug can invalidate entire competition
6. **Document everything**—advanced tactics are hard to reproduce
