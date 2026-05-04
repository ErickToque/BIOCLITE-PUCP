"""BIOCLITE-PUCP: Transfer Learning for Bradykinesia Detection

This package provides tools for:
- Loading and preprocessing BIOCLITE smartwatch data
- Extracting time and frequency domain features
- Training machine learning models for bradykinesia detection
- Transfer learning from clinical to home settings
"""

__version__ = "1.0.0"
__author__ = "BIOCLITE Research Group"

# Data loading and preprocessing
from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

# Utilities
from src.utils.helpers import set_seed, get_device, compute_metrics

# Visualization
from src.visualization.results import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# Models (lazy import to avoid circular dependencies)
def get_model(model_name):
    """Lazy import for models"""
    if model_name == 'CNN1D':
        from src.models.deep_learning import CNN1D
        return CNN1D
    elif model_name == 'BiLSTM':
        from src.models.deep_learning import BiLSTM
        return BiLSTM
    elif model_name == 'SimpleCNN':
        from src.models.deep_learning import SimpleCNN
        return SimpleCNN
    else:
        raise ValueError(f"Unknown model: {model_name}")

__all__ = [
    'BIOCLITEDataset',
    'IMUPreprocessor',
    'set_seed',
    'get_device',
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_importance',
    'get_model',
]
