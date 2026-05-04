"""
Utility functions for BIOCLITE-PUCP - CSV Version
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from datetime import datetime

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def save_model(model, path):
    """Save PyTorch model"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load PyTorch model"""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_subject_split(df, subject_col='subject_id', train_ratio=0.7, val_ratio=0.15, random_state=42):
    """Split by subject for subject-independent evaluation"""
    np.random.seed(random_state)
    
    subjects = df[subject_col].unique()
    np.random.shuffle(subjects)
    
    n_train = int(len(subjects) * train_ratio)
    n_val = int(len(subjects) * val_ratio)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    return train_subjects, val_subjects, test_subjects

def print_dataset_info(df):
    """Print comprehensive dataset information"""
    print("=" * 60)
    print("BIOCLITE DATASET INFO")
    print("=" * 60)
    
    print(f"\n📊 Basic Stats:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Sampling rate: 50 Hz")
    print(f"  Duration: {len(df) / 50 / 3600:.1f} hours")
    
    print(f"\n👥 Participants:")
    print(f"  Total: {df['subject_id'].nunique()}")
    print(f"  Parkinson (Grupo=1): {df[df['Grupo_sesion']==1]['subject_id'].nunique()}")
    print(f"  Healthy (Grupo=0): {df[df['Grupo_sesion']==0]['subject_id'].nunique()}")
    
    print(f"\n🎯 UPDRS Distribution:")
    for updrs in sorted(df['UPDRS'].unique()):
        count = (df['UPDRS'] == updrs).sum()
        pct = count / len(df) * 100
        label = "Not available" if updrs == 99 else f"Score {updrs}"
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    print(f"\n🎮 Context Distribution:")
    for ctx in sorted(df['Contexto_sesion'].unique()):
        ctx_name = {0: "Home (unsupervised)", 1: "Clinic initial", 2: "Clinic final"}[ctx]
        count = (df['Contexto_sesion'] == ctx).sum()
        pct = count / len(df) * 100
        print(f"  {ctx_name}: {count:,} ({pct:.1f}%)")
    
    print(f"\n🏋️ Exercise Distribution:")
    for ex in sorted(df['Ejercicio'].unique()):
        count = (df['Ejercicio'] == ex).sum()
        pct = count / len(df) * 100
        print(f"  Exercise {ex}: {count:,} ({pct:.1f}%)")