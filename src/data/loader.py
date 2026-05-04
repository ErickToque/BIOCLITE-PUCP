"""
Data loader for BIOCLITE dataset - CSV Version
40 participants (24 PD, 16 healthy) with supervised/unsupervised contexts
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

class BIOCLITEDataset:
    """Load and process BIOCLITE smartwatch dataset from CSV"""
    
    def __init__(self, data_path='data/raw/BIOCLITE_data_v2.csv', fs=50):
        self.data_path = data_path
        self.fs = fs
        self.df = None
        self.all_sessions = []
        
    def load_data(self):
        """Load the CSV file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Create unique subject ID if not exists
        if 'subject_id' not in self.df.columns:
            self.df['subject_id'] = self.df['Grupo_sesion'].astype(str) + '_' + self.df['Participante_sesion'].astype(str)
        
        print(f"✅ Loaded {len(self.df):,} rows")
        print(f"   Subjects: {self.df['subject_id'].nunique()} (PD: {self.df[self.df['Grupo_sesion']==1]['subject_id'].nunique()}, Healthy: {self.df[self.df['Grupo_sesion']==0]['subject_id'].nunique()})")
        print(f"   Sessions: {self.df['Sesion'].nunique()}")
        print(f"   Exercises: {sorted(self.df['Ejercicio'].unique())}")
        
        return self.df
    
    def get_exercise_data(self, exercise_num, context=None):
        """Extract data for a specific exercise"""
        df_ex = self.df[self.df['Ejercicio'] == exercise_num].copy()
        
        if context is not None:
            df_ex = df_ex[df_ex['Contexto_sesion'] == context]
        
        return df_ex
    
    def extract_windows(self, df_ex, window_size=100, step_size=50):
        """Extract sliding windows from exercise data"""
        X, y, groups = [], [], []
        
        for session in df_ex['Sesion'].unique():
            subset = df_ex[df_ex['Sesion'] == session]
            
            # Extract IMU signals
            acc_cols = ['Acc_X', 'Acc_Y', 'Acc_Z']
            gyro_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
            
            acc = subset[acc_cols].values
            gyro = subset[gyro_cols].values
            
            if len(acc) < window_size:
                continue
            
            # Create sliding windows
            for i in range(0, len(acc) - window_size, step_size):
                acc_window = acc[i:i+window_size]
                gyro_window = gyro[i:i+window_size]
                
                # Combine acc and gyro
                window = np.hstack([acc_window, gyro_window])
                
                X.append(window)
                
                # Label: presence of symptom (UPDRS > 0 and != 99)
                label = subset['UPDRS'].iloc[0]
                y.append(1 if label not in [0, 99] else 0)
                
                # Group by subject
                groups.append(subset['subject_id'].iloc[0])
        
        return np.array(X), np.array(y), np.array(groups)


class IMUDataset(Dataset):
    """PyTorch Dataset for IMU windows"""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def create_dataloaders(X, y, groups, subject_id=None, batch_size=32, train_ratio=0.8, random_state=42):
    """Create train/test dataloaders with subject-aware splitting"""
    np.random.seed(random_state)
    
    unique_subjects = np.unique(groups)
    n_train = int(len(unique_subjects) * train_ratio)
    
    train_subjects = np.random.choice(unique_subjects, n_train, replace=False)
    test_subjects = [s for s in unique_subjects if s not in train_subjects]
    
    train_idx = [i for i, g in enumerate(groups) if g in train_subjects]
    test_idx = [i for i, g in enumerate(groups) if g in test_subjects]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Scale features
    scaler = RobustScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    train_dataset = IMUDataset(X_train_scaled, y_train)
    test_dataset = IMUDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler


if __name__ == "__main__":
    # Test the loader
    loader = BIOCLITEDataset()
    df = loader.load_data()
    
    # Test exercise 6 (bradykinesia)
    df_ej6 = loader.get_exercise_data(exercise_num=6)
    print(f"\nExercise 6: {len(df_ej6)} samples")
    
    X, y, groups = loader.extract_windows(df_ej6, window_size=100, step_size=50)
    print(f"Windows: {X.shape}")
    print(f"Classes: {np.bincount(y)}")
    print(f"Subjects: {np.unique(groups)}")