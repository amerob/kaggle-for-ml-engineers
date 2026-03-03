# Part 4: Model Architecture Deep Dives

## GBDT: The Workhorse of Tabular ML

Gradient Boosted Decision Trees (GBDT) remain the dominant architecture for tabular data competitions. LightGBM, XGBoost, and CatBoost each have strengths—understanding when to use each is crucial.

### GBDT Comparison Table

| Feature | LightGBM | XGBoost | CatBoost |
|---------|----------|---------|----------|
| **Training Speed** | Fastest | Medium | Medium |
| **Memory Usage** | Lowest | Medium | Highest |
| **Categorical Support** | Requires encoding | Requires encoding | Native |
| **GPU Support** | Yes | Yes | Yes |
| **Default Performance** | Excellent | Good | Excellent |
| **Hyperparameter Sensitivity** | Medium | High | Low |
| **Best For** | Large datasets | Fine control | Categorical-heavy |

### When to Use Each

```
                    Dataset Size?
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    Small (<100k)   Medium       Large (>1M)
        │              │              │
    CatBoost    LightGBM/XGBoost   LightGBM
    (default)    (tune for best)   (default)
        
                    Categorical %?
                       │
              ┌────────┴────────┐
              │                 │
           High (>30%)       Low (<30%)
              │                 │
          CatBoost         LightGBM
```

---

## LightGBM Configuration Template

### Production-Ready LightGBM Configuration

```python
import lightgbm as lgb
import numpy as np

class LightGBMConfig:
    """
    Production-ready LightGBM configurations for different scenarios.
    """
    
    # Base configuration (good starting point)
    BASE_CONFIG = {
        'objective': 'binary',  # or 'regression', 'multiclass'
        'metric': 'auc',        # or 'rmse', 'mlogloss'
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 10000,  # Use early stopping
    }
    
    # Fast iteration config (for experimentation)
    FAST_CONFIG = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 20,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 1000,
    }
    
    # High-performance config (for final submissions)
    HIGH_PERF_CONFIG = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 20000,
    }
    
    # GPU config
    GPU_CONFIG = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 10000,
    }
    
    @classmethod
    def get_config(cls, scenario='base', objective='binary', metric='auc'):
        """
        Get configuration for specific scenario.
        
        Args:
            scenario: 'fast', 'base', 'high_perf', or 'gpu'
            objective: 'binary', 'regression', or 'multiclass'
            metric: Evaluation metric
        """
        configs = {
            'fast': cls.FAST_CONFIG,
            'base': cls.BASE_CONFIG,
            'high_perf': cls.HIGH_PERF_CONFIG,
            'gpu': cls.GPU_CONFIG,
        }
        
        config = configs.get(scenario, cls.BASE_CONFIG).copy()
        config['objective'] = objective
        config['metric'] = metric
        
        if objective == 'multiclass':
            config['num_class'] = None  # Set based on data
            
        return config


# Usage example
def train_lightgbm(X_train, y_train, X_val, y_val, scenario='base'):
    """Train LightGBM with proper configuration."""
    
    config = LightGBMConfig.get_config(scenario=scenario)
    
    model = lgb.LGBMClassifier(**config)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model
```

### LightGBM Hyperparameter Guide

| Parameter | Default | Range | Effect | Tuning Strategy |
|-----------|---------|-------|--------|-----------------|
| `num_leaves` | 31 | 10-150 | Model complexity | Increase for complex patterns |
| `learning_rate` | 0.05 | 0.01-0.3 | Step size | Lower with more trees |
| `feature_fraction` | 0.8 | 0.5-1.0 | Column sampling | Lower to prevent overfitting |
| `bagging_fraction` | 0.8 | 0.5-1.0 | Row sampling | Lower for large datasets |
| `min_child_samples` | 20 | 5-100 | Leaf size | Increase to reduce overfitting |
| `reg_alpha` | 0 | 0-1.0 | L1 regularization | Increase for sparse features |
| `reg_lambda` | 0 | 0-1.0 | L2 regularization | Increase for correlated features |

---

## CatBoost for Categorical Data

CatBoost's native categorical handling makes it the go-to choice for categorical-heavy datasets.

### CatBoost Best Practices

```python
import catboost as cb

class CatBoostTrainer:
    """
    Optimized CatBoost training pipeline.
    """
    
    def __init__(self, categorical_features, task_type='CPU'):
        self.categorical_features = categorical_features
        self.task_type = task_type
        self.model = None
        
    def get_params(self, iterations=10000, learning_rate=0.05):
        """Get optimized CatBoost parameters."""
        return {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'early_stopping_rounds': 100,
            'use_best_model': True,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'task_type': self.task_type,
            'verbose': 100,
            'cat_features': self.categorical_features,
            # Ordered boosting for small datasets
            'boosting_type': 'Ordered' if self.task_type == 'CPU' else 'Plain',
        }
    
    def train(self, X_train, y_train, X_val, y_val, 
              iterations=10000, learning_rate=0.05):
        """Train CatBoost model."""
        
        params = self.get_params(iterations, learning_rate)
        
        # Create pools
        train_pool = cb.Pool(
            X_train, y_train,
            cat_features=self.categorical_features
        )
        val_pool = cb.Pool(
            X_val, y_val,
            cat_features=self.categorical_features
        )
        
        # Train
        self.model = cb.CatBoostClassifier(**params)
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
        )
        
        return self.model
    
    def get_feature_importance(self, importance_type='PredictionValuesChange'):
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.get_feature_importance(
            type=importance_type
        )
        
        return dict(zip(self.model.feature_names_, importance))


# Usage example
# trainer = CatBoostTrainer(
#     categorical_features=['category1', 'category2', 'category3'],
#     task_type='GPU'
# )
# model = trainer.train(X_train, y_train, X_val, y_val)
```

---

## NLP: Transformers & Beyond

### Modern NLP Stack

```
┌─────────────────────────────────────────────────────┐
│                 Modern NLP Pipeline                  │
├─────────────────────────────────────────────────────┤
│  1. Tokenization (AutoTokenizer)                    │
│  2. Pre-trained Model (BERT/RoBERTa/DeBERTa)        │
│  3. Fine-tuning with LoRA/QLoRA                     │
│  4. Classification Head                             │
│  5. Inference Optimization                          │
└─────────────────────────────────────────────────────┘
```

### LoRA Fine-Tuning Implementation

```python
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

class LoRATextClassifier:
    """
    Text classifier with LoRA fine-tuning.
    
    Efficient fine-tuning of large language models.
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-base', 
                 num_labels=2, max_length=512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
    def setup_lora(self, r=16, lora_alpha=32, lora_dropout=0.1):
        """
        Setup LoRA configuration.
        
        Args:
            r: LoRA rank (lower = more compression)
            lora_alpha: Scaling factor
            lora_dropout: Dropout for LoRA layers
        """
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type='single_label_classification'
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=['query_proj', 'key_proj', 'value_proj', 'dense'],
            lora_dropout=lora_dropout,
            bias='none',
            task_type=TaskType.SEQ_CLS
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def tokenize_function(self, examples):
        """Tokenize text data."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        if self.num_labels == 2:
            # Binary classification
            probs = torch.nn.functional.softmax(
                torch.tensor(predictions), dim=-1
            )[:, 1].numpy()
            
            return {
                'accuracy': accuracy_score(labels, predictions.argmax(-1)),
                'auc': roc_auc_score(labels, probs)
            }
        else:
            # Multi-class
            return {
                'accuracy': accuracy_score(labels, predictions.argmax(-1))
            }
    
    def train(self, train_dataset, val_dataset, output_dir='./results',
              num_epochs=3, batch_size=16, learning_rate=2e-4):
        """Train the model."""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_ratio=0.1,
            weight_decay=0.01,
            learning_rate=learning_rate,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_auc' if self.num_labels == 2 else 'eval_accuracy',
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            report_to='none'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        
        return trainer
    
    def predict(self, texts):
        """Make predictions on new texts."""
        self.model.eval()
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return probs.cpu().numpy()


# Model selection guide for NLP competitions
NLP_MODEL_GUIDE = {
    'general_classification': {
        'model': 'microsoft/deberta-v3-base',
        'size': '86M params',
        'pros': 'Fast, good baseline',
        'cons': 'May underfit complex tasks'
    },
    'complex_reasoning': {
        'model': 'microsoft/deberta-v3-large',
        'size': '304M params',
        'pros': 'Best performance',
        'cons': 'Slower, needs more GPU'
    },
    'long_documents': {
        'model': 'allenai/longformer-base-4096',
        'size': '149M params',
        'pros': 'Handles long texts',
        'cons': 'Slower than DeBERTa'
    },
    'multilingual': {
        'model': 'FacebookAI/xlm-roberta-large',
        'size': '550M params',
        'pros': 'Multilingual',
        'cons': 'Very large, slow'
    }
}
```

---

## Computer Vision: CNNs & Vision Transformers

### Modern CV Architecture Selection

```
                    Dataset Size?
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     Small           Medium         Large
    (<10k)        (10k-100k)      (>100k)
        │              │              │
    ResNet50      EfficientNet     Swin/ViT
    (pretrained)   (B3-B5)        (fine-tuned)
```

### timm-Based Training Pipeline

```python
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

class TimmClassifier:
    """
    Image classifier using timm models.
    """
    
    def __init__(self, model_name='efficientnet_b3', 
                 num_classes=2, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self, dropout=0.2):
        """Build model with custom head."""
        self.model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=0,  # Remove default head
            drop_rate=dropout
        )
        
        # Get number of features
        num_features = self.model.num_features
        
        # Custom classification head
        self.model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, self.num_classes)
        )
        
        self.model = self.model.to(self.device)
        
        return self.model
    
    def get_transforms(self, img_size=224, is_training=True):
        """Get image transforms."""
        if is_training:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader, criterion):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), correct / total, np.array(all_probs)


# Model selection for CV competitions
CV_MODEL_GUIDE = {
    'fast_baseline': {
        'model': 'resnet50',
        'img_size': 224,
        'batch_size': 64,
        'pros': 'Fast, reliable',
        'cons': 'Not state-of-art'
    },
    'balanced': {
        'model': 'efficientnet_b3',
        'img_size': 300,
        'batch_size': 32,
        'pros': 'Good speed/accuracy',
        'cons': 'None major'
    },
    'high_accuracy': {
        'model': 'efficientnet_b5',
        'img_size': 456,
        'batch_size': 16,
        'pros': 'Best accuracy',
        'cons': 'Slow, memory-heavy'
    },
    'transformer': {
        'model': 'swin_base_patch4_window7_224',
        'img_size': 224,
        'batch_size': 16,
        'pros': 'SOTA performance',
        'cons': 'Needs more data'
    }
}
```

---

## Time Series: Specialized Patterns

### Time Series Feature Engineering

```python
import pandas as pd
import numpy as np

class TimeSeriesFeatureEngineer:
    """
    Feature engineering for time series data.
    """
    
    def __init__(self, datetime_col, target_col):
        self.datetime_col = datetime_col
        self.target_col = target_col
        
    def create_lag_features(self, df, lags=[1, 7, 14, 30]):
        """Create lag features."""
        df = df.copy()
        
        for lag in lags:
            df[f'{self.target_col}_lag_{lag}'] = df.groupby(
                'group_id'
            )[self.target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, windows=[7, 14, 30]):
        """Create rolling window statistics."""
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'{self.target_col}_rolling_mean_{window}'] = (
                df.groupby('group_id')[self.target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Rolling std
            df[f'{self.target_col}_rolling_std_{window}'] = (
                df.groupby('group_id')[self.target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            
            # Rolling min/max
            df[f'{self.target_col}_rolling_min_{window}'] = (
                df.groupby('group_id')[self.target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).min())
            )
            
            df[f'{self.target_col}_rolling_max_{window}'] = (
                df.groupby('group_id')[self.target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).max())
            )
            
            # Expanding mean
            df[f'{self.target_col}_expanding_mean'] = (
                df.groupby('group_id')[self.target_col]
                .transform(lambda x: x.expanding().mean())
            )
        
        return df
    
    def create_datetime_features(self, df):
        """Create datetime-based features."""
        df = df.copy()
        
        # Ensure datetime
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        
        # Basic features
        df['year'] = df[self.datetime_col].dt.year
        df['month'] = df[self.datetime_col].dt.month
        df['day'] = df[self.datetime_col].dt.day
        df['dayofweek'] = df[self.datetime_col].dt.dayofweek
        df['dayofyear'] = df[self.datetime_col].dt.dayofyear
        df['weekofyear'] = df[self.datetime_col].dt.isocalendar().week
        df['quarter'] = df[self.datetime_col].dt.quarter
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Is weekend
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Is month start/end
        df['is_month_start'] = df[self.datetime_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[self.datetime_col].dt.is_month_end.astype(int)
        
        return df
    
    def create_all_features(self, df, lag_lags=[1, 7, 14, 30], 
                           rolling_windows=[7, 14, 30]):
        """Create all time series features."""
        df = self.create_lag_features(df, lag_lags)
        df = self.create_rolling_features(df, rolling_windows)
        df = self.create_datetime_features(df)
        
        return df
```

### Time Series Model: Temporal Fusion Transformer

```python
# For advanced time series, consider Temporal Fusion Transformer
# Requires: pip install pytorch-forecasting

"""
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

def create_tft_model(training_data, max_prediction_length, max_encoder_length):
    '''
    Create Temporal Fusion Transformer for multi-horizon forecasting.
    '''
    # Define dataset
    training = TimeSeriesDataSet(
        training_data,
        time_idx='time_idx',
        target='target',
        group_ids=['group_id'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['category'],
        time_varying_known_reals=['time_idx', 'month', 'dayofweek'],
        time_varying_unknown_reals=['target'],
        target_normalizer=GroupNormalizer(
            groups=['group_id'], transformation='softplus'
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=160,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=160,
        output_size=7,  # Quantile output
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    return tft, training
"""
```

---

## Model Selection Checklist

- [ ] Identify data type (tabular/CV/NLP/time series)
- [ ] Estimate dataset size and complexity
- [ ] Check for categorical features (>30% → CatBoost)
- [ ] Check compute constraints (GPU available?)
- [ ] Start with strong baseline (LightGBM/ResNet50)
- [ ] Experiment with 2-3 architectures
- [ ] Validate with proper CV
- [ ] Consider ensemble potential
- [ ] Document hyperparameter choices
- [ ] Save model artifacts with version

---

## Key Takeaways

1. **GBDT dominates tabular**—master LightGBM, XGBoost, CatBoost
2. **CatBoost for categorical-heavy data**—native handling is superior
3. **Transformers for NLP**—LoRA makes fine-tuning accessible
4. **timm for CV**—pretrained models + proper augmentation win
5. **Time series needs special handling**—lags, rolling features, temporal CV
6. **Start simple, then scale**—baseline first, complexity later
7. **Match model to data size**—don't use ViT on 1k images
