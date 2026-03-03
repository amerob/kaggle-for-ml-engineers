# Part 7: Production Reality

## Bridging the Gap

This chapter addresses the critical translation from competition-winning techniques to production-ready systems. While Kaggle rewards pure predictive performance, production systems must balance accuracy with latency, cost, maintainability, and reliability.

---

## Scaling Challenges

### The Latency-Accuracy Tradeoff

```
Accuracy
    │
    │         ╭─────── Competition Optimum
    │        ╱
    │       ╱
    │      ╱    ╭─────── Production Optimum
    │     ╱    ╱
    │    ╱    ╱
    │   ╱    ╱
    │  ╱    ╱
    │ ╱    ╱
    │╱____╱________________
     │                    Latency
     Low        High
```

### Production Latency Requirements

| Use Case | p50 Latency | p99 Latency | Throughput | Accuracy Requirement |
|----------|-------------|-------------|------------|---------------------|
| **Search Ranking** | <10ms | <50ms | 100k QPS | Good (0.75 AUC) |
| **Ad Click Prediction** | <5ms | <20ms | 1M QPS | Good (0.70 AUC) |
| **Fraud Detection** | <50ms | <200ms | 10k QPS | Excellent (0.90 AUC) |
| **Recommendation** | <20ms | <100ms | 50k QPS | Good (0.75 AUC) |
| **Credit Scoring** | <100ms | <500ms | 1k QPS | Excellent (0.85 AUC) |
| **Batch Scoring** | <1min | <5min | 1M/hr | Excellent (0.90 AUC) |

### Scaling Dimensions

```python
class ScalingAnalyzer:
    """
    Analyze model scaling requirements.
    """
    
    def __init__(self, model, sample_input):
        self.model = model
        self.sample_input = sample_input
        
    def measure_latency(self, n_runs=100):
        """Measure inference latency."""
        import time
        
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            _ = self.model.predict(self.sample_input)
            latencies.append((time.time() - start) * 1000)
        
        return {
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'mean': np.mean(latencies),
            'std': np.std(latencies)
        }
    
    def estimate_throughput(self, batch_size=1):
        """Estimate maximum throughput."""
        import time
        
        n_samples = 1000
        start = time.time()
        
        for i in range(0, n_samples, batch_size):
            batch = self.sample_input[:batch_size]
            _ = self.model.predict(batch)
        
        elapsed = time.time() - start
        throughput = n_samples / elapsed
        
        return {
            'throughput_qps': throughput,
            'batch_size': batch_size,
            'total_time_sec': elapsed
        }
    
    def memory_footprint(self):
        """Estimate model memory footprint."""
        import sys
        import joblib
        import io
        
        # Serialize model
        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)
        model_size_mb = buffer.tell() / (1024 * 1024)
        
        return {
            'serialized_size_mb': model_size_mb,
            'estimated_ram_mb': model_size_mb * 3  # Rough estimate
        }
    
    def full_analysis(self):
        """Run complete scaling analysis."""
        return {
            'latency_ms': self.measure_latency(),
            'throughput': self.estimate_throughput(),
            'memory': self.memory_footprint()
        }
```

---

## Model Compression Techniques

### Compression Methods Comparison

| Method | Compression Ratio | Accuracy Loss | Implementation Complexity | Use Case |
|--------|-------------------|---------------|--------------------------|----------|
| **Quantization (INT8)** | 4x | 0-1% | Low | Most models |
| **Pruning** | 2-10x | 0-2% | Medium | Overparameterized models |
| **Knowledge Distillation** | 10-100x | 1-3% | High | Large → Small |
| **ONNX Conversion** | 1-2x | 0% | Low | Cross-platform deployment |
| **TensorRT** | 2-5x | 0% | Medium | NVIDIA GPUs |

### ModelCompressor Implementation

```python
import numpy as np
import joblib
import onnx
import torch
import torch.onnx

class ModelCompressor:
    """
    Model compression pipeline for production deployment.
    """
    
    def __init__(self, model, model_type='sklearn'):
        self.original_model = model
        self.model_type = model_type
        self.compressed_model = None
        self.compression_stats = {}
        
    def quantize(self, X_sample, n_bits=8):
        """
        Quantize model to lower precision.
        
        Args:
            X_sample: Sample data for calibration
            n_bits: Number of bits (8 or 16)
        """
        if self.model_type == 'sklearn':
            # For tree-based models, use histogram binning
            from sklearn.ensemble import RandomForestClassifier
            
            # Quantize feature thresholds
            model_copy = joblib.loads(joblib.dumps(self.original_model))
            
            if hasattr(model_copy, 'estimators_'):
                for tree in model_copy.estimators_:
                    if hasattr(tree, 'tree_'):
                        tree_ = tree.tree_
                        # Quantize thresholds
                        thresholds = tree_.threshold
                        # Simple uniform quantization
                        min_val, max_val = thresholds.min(), thresholds.max()
                        n_bins = 2 ** n_bits
                        bins = np.linspace(min_val, max_val, n_bins)
                        quantized = np.digitize(thresholds, bins)
                        tree_.threshold = bins[quantized - 1]
            
            self.compressed_model = model_copy
            
        elif self.model_type == 'pytorch':
            # Dynamic quantization for PyTorch
            self.compressed_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        self.compression_stats['quantization'] = {
            'n_bits': n_bits,
            'method': 'dynamic' if self.model_type == 'pytorch' else 'uniform'
        }
        
        return self.compressed_model
    
    def prune(self, X_val, y_val, sparsity=0.3):
        """
        Prune model to remove less important parameters.
        
        Args:
            X_val: Validation data
            y_val: Validation labels
            sparsity: Target sparsity ratio (0-1)
        """
        if self.model_type == 'pytorch':
            import torch.nn.utils.prune as prune
            
            model_copy = joblib.loads(joblib.dumps(self.original_model))
            
            # Prune linear layers
            for name, module in model_copy.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(
                        module, 
                        name='weight', 
                        amount=sparsity
                    )
            
            self.compressed_model = model_copy
            
        elif self.model_type == 'sklearn':
            # Feature importance-based pruning for tree models
            from sklearn.feature_selection import SelectFromModel
            
            selector = SelectFromModel(
                self.original_model, 
                threshold='median'
            )
            selector.fit(X_val, y_val)
            
            self.compression_stats['pruning'] = {
                'sparsity': sparsity,
                'n_features_selected': selector.get_support().sum()
            }
            
            return selector
        
        self.compression_stats['pruning'] = {'sparsity': sparsity}
        
        return self.compressed_model
    
    def convert_to_onnx(self, input_sample, output_path):
        """
        Convert model to ONNX format.
        
        Args:
            input_sample: Sample input for tracing
            output_path: Path to save ONNX model
        """
        if self.model_type == 'pytorch':
            torch.onnx.export(
                self.original_model,
                input_sample,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            self.compression_stats['onnx'] = {
                'path': output_path,
                'opset_version': 11
            }
            
        elif self.model_type == 'sklearn':
            # Use skl2onnx for sklearn models
            from skl2onnx import convert_lightgbm
            from skl2onnx.common.data_types import FloatTensorType
            
            initial_type = [('float_input', FloatTensorType([None, input_sample.shape[1]]))]
            onnx_model = convert_lightgbm(self.original_model, initial_types=initial_type)
            
            with open(output_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
        
        return output_path
    
    def benchmark_compression(self, X_test, y_test):
        """
        Benchmark original vs compressed model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Benchmark results
        """
        import time
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        results = {
            'original': {},
            'compressed': {}
        }
        
        # Benchmark original
        start = time.time()
        if hasattr(self.original_model, 'predict_proba'):
            orig_preds = self.original_model.predict_proba(X_test)[:, 1]
            results['original']['auc'] = roc_auc_score(y_test, orig_preds)
        orig_preds_class = self.original_model.predict(X_test)
        results['original']['accuracy'] = accuracy_score(y_test, orig_preds_class)
        results['original']['inference_time_ms'] = (time.time() - start) * 1000 / len(X_test)
        
        # Benchmark compressed
        if self.compressed_model is not None:
            start = time.time()
            if hasattr(self.compressed_model, 'predict_proba'):
                comp_preds = self.compressed_model.predict_proba(X_test)[:, 1]
                results['compressed']['auc'] = roc_auc_score(y_test, comp_preds)
            comp_preds_class = self.compressed_model.predict(X_test)
            results['compressed']['accuracy'] = accuracy_score(y_test, comp_preds_class)
            results['compressed']['inference_time_ms'] = (time.time() - start) * 1000 / len(X_test)
            
            # Calculate degradation
            results['accuracy_degradation'] = (
                results['original']['accuracy'] - results['compressed']['accuracy']
            )
        
        return results


# Usage example
"""
# Create compressor
compressor = ModelCompressor(model, model_type='pytorch')

# Apply quantization
quantized_model = compressor.quantize(X_sample, n_bits=8)

# Convert to ONNX
compressor.convert_to_onnx(torch.randn(1, input_dim), 'model.onnx')

# Benchmark
results = compressor.benchmark_compression(X_test, y_test)
print(f"Accuracy degradation: {results['accuracy_degradation']:.4f}")
"""
```

---

## Monitoring & MLOps Gaps

### Production Monitoring Requirements

```python
import numpy as np
from datetime import datetime, timedelta
import json

class ProductionMonitor:
    """
    Production model monitoring system.
    
    Tracks performance, data drift, and system health.
    """
    
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.metrics_history = []
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'latency_p99': 100,  # ms
            'error_rate': 0.01,
            'data_drift': 0.1
        }
        
    def log_prediction(self, features, prediction, actual=None, 
                       latency_ms=None, timestamp=None):
        """
        Log a single prediction.
        
        Args:
            features: Input features
            prediction: Model prediction
            actual: Ground truth (if available)
            latency_ms: Inference latency
            timestamp: Prediction timestamp
        """
        log_entry = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'prediction': float(prediction),
            'latency_ms': latency_ms,
        }
        
        if actual is not None:
            log_entry['actual'] = float(actual)
            log_entry['error'] = abs(float(prediction) - float(actual))
        
        # Feature statistics
        log_entry['feature_stats'] = {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features))
        }
        
        self.metrics_history.append(log_entry)
        
        return log_entry
    
    def calculate_drift(self, reference_stats, current_stats):
        """
        Calculate data drift between reference and current.
        
        Args:
            reference_stats: Reference distribution statistics
            current_stats: Current distribution statistics
            
        Returns:
            Drift metrics
        """
        # Population Stability Index (PSI)
        psi = 0
        for bin_name in reference_stats:
            ref_pct = reference_stats[bin_name]
            curr_pct = current_stats.get(bin_name, 0.001)  # Small constant to avoid div by zero
            
            psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
        
        return {
            'psi': psi,
            'drift_detected': psi > 0.25  # Standard threshold
        }
    
    def generate_report(self, window_hours=24):
        """
        Generate monitoring report.
        
        Args:
            window_hours: Time window for report
            
        Returns:
            Report dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        recent_logs = [
            log for log in self.metrics_history
            if datetime.fromisoformat(log['timestamp']) > cutoff_time
        ]
        
        if not recent_logs:
            return {'error': 'No data in window'}
        
        # Calculate metrics
        latencies = [log['latency_ms'] for log in recent_logs if log['latency_ms']]
        errors = [log['error'] for log in recent_logs if 'error' in log]
        
        report = {
            'window_hours': window_hours,
            'total_predictions': len(recent_logs),
            'latency': {
                'p50_ms': np.percentile(latencies, 50) if latencies else None,
                'p95_ms': np.percentile(latencies, 95) if latencies else None,
                'p99_ms': np.percentile(latencies, 99) if latencies else None,
            },
            'alerts': []
        }
        
        if errors:
            report['mae'] = np.mean(errors)
            report['rmse'] = np.sqrt(np.mean([e**2 for e in errors]))
        
        # Check thresholds
        if report['latency']['p99_ms'] and report['latency']['p99_ms'] > self.alert_thresholds['latency_p99']:
            report['alerts'].append({
                'type': 'latency',
                'message': f"p99 latency ({report['latency']['p99_ms']:.1f}ms) exceeds threshold"
            })
        
        return report
    
    def check_health(self):
        """Quick health check."""
        report = self.generate_report(window_hours=1)
        
        return {
            'healthy': len(report.get('alerts', [])) == 0,
            'alerts': report.get('alerts', []),
            'last_prediction': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
        }
```

### MLOps Integration Checklist

| Component | Competition | Production | Gap |
|-----------|-------------|------------|-----|
| **Version Control** | Git for code | Git + MLflow/DVC for models/data | Significant |
| **Experiment Tracking** | Spreadsheets/notebooks | MLflow, Weights & Biases | Significant |
| **Model Registry** | Manual | MLflow Model Registry, S3 | Significant |
| **CI/CD** | Manual runs | Automated pipelines | Major |
| **Monitoring** | Post-hoc | Real-time dashboards | Major |
| **A/B Testing** | CV comparison | Shadow deployment, traffic splitting | Major |
| **Retraining** | Manual | Automated triggers | Significant |

### Production Deployment Checklist

```python
PRODUCTION_DEPLOYMENT_CHECKLIST = {
    'pre_deployment': [
        'Model validated on holdout set',
        'Latency benchmarks meet requirements',
        'Memory footprint acceptable',
        'Model serialized and versioned',
        'Feature pipeline tested end-to-end',
        'Fallback model prepared',
        'Rollback plan documented',
    ],
    'deployment': [
        'Canary deployment (5% traffic)',
        'Shadow deployment comparison',
        'Error rate monitoring',
        'Latency monitoring',
        'Prediction distribution monitoring',
    ],
    'post_deployment': [
        'A/B test results analyzed',
        'Business metrics tracked',
        'Model performance logged',
        'Alert thresholds configured',
        'Documentation updated',
    ]
}
```

---

## Production Adaptation Patterns

### From Competition to Production

| Competition Pattern | Production Adaptation | Rationale |
|---------------------|----------------------|-----------|
| 20-model ensemble | 3-model ensemble | Latency constraints |
| Complex features | Pre-computed features | Real-time serving |
| GPU training | CPU inference | Cost optimization |
| Batch predictions | Real-time API | User experience |
| Weekly retraining | Continuous learning | Freshness |

### Feature Store Integration

```python
class FeatureStoreAdapter:
    """
    Adapter for production feature store integration.
    
    Separates online (real-time) and offline (batch) features.
    """
    
    def __init__(self, feature_store_client):
        self.client = feature_store_client
        self.online_features = []
        self.offline_features = []
        
    def register_features(self, feature_name, feature_type='offline'):
        """Register feature with store."""
        if feature_type == 'online':
            self.online_features.append(feature_name)
        else:
            self.offline_features.append(feature_name)
    
    def get_online_features(self, entity_id):
        """Fetch real-time features."""
        return self.client.get_online_features(
            entity_id, 
            self.online_features
        )
    
    def get_offline_features(self, entity_ids):
        """Fetch batch features."""
        return self.client.get_offline_features(
            entity_ids,
            self.offline_features
        )
    
    def compute_feature_vector(self, entity_id, raw_features):
        """
        Compute complete feature vector.
        
        Combines online and offline features.
        """
        online = self.get_online_features(entity_id)
        offline = self.get_offline_features([entity_id])
        
        # Combine
        feature_vector = {**online, **offline, **raw_features}
        
        return feature_vector
```

---

## Key Takeaways

1. **Latency is a constraint**—not just accuracy
2. **Model compression is essential**—quantization, pruning, distillation
3. **Monitoring is non-negotiable**—drift, performance, latency
4. **MLOps gaps are real**—plan for significant adaptation
5. **Feature stores separate concerns**—online vs offline features
6. **Start with canary deployments**—validate before full rollout
7. **Document everything**—production systems live for years
