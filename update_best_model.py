#!/usr/bin/env python3
"""
Update Random Forest model with best hyperparameters from grid search
"""

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*60)
print("🎯 ACTUALIZANDO MODELO CON MEJORES HIPERPARÁMETROS")
print("="*60)

# Best parameters from grid search
BEST_PARAMS = {
    'n_estimators': 50,
    'max_depth': 4,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.3,
}

print(f"\n📊 Mejores parámetros:")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")

# Load data
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

# Extract features
WINDOW_SIZE = 100
STEP_SIZE = 100
preprocessor = IMUPreprocessor(fs=50)

X, y, groups = [], [], []

for session in df_clinic['Sesion'].unique():
    df_ses = df_clinic[df_clinic['Sesion'] == session]
    acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
    
    if len(acc) < WINDOW_SIZE:
        continue
    
    for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
        acc_window = acc[i:i+WINDOW_SIZE]
        gyro_window = gyro[i:i+WINDOW_SIZE]
        
        features = preprocessor.extract_features(acc_window, gyro_window)
        X.append(list(features.values()))
        
        updrs = df_ses['UPDRS'].iloc[0]
        label = 1 if updrs not in [0, 99] else 0
        y.append(label)
        groups.append(df_ses['subject_id'].iloc[0])

X = np.array(X)
y = np.array(y)

print(f"\n📊 Dataset: {X.shape[0]} ventanas, {X.shape[1]} features")

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y==0)-1))
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

print(f"  Balanceado: {X_balanced.shape[0]} muestras")

# Train with best parameters
rf = RandomForestClassifier(
    n_estimators=BEST_PARAMS['n_estimators'],
    max_depth=BEST_PARAMS['max_depth'],
    min_samples_split=BEST_PARAMS['min_samples_split'],
    min_samples_leaf=BEST_PARAMS['min_samples_leaf'],
    max_features=BEST_PARAMS['max_features'],
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_balanced, y_balanced)

# Save
joblib.dump(rf, 'models/saved/best_random_forest_optimized.pkl')
joblib.dump(scaler, 'models/saved/scaler_optimized.pkl')

print("\n✅ Modelo optimizado guardado:")
print("  • models/saved/best_random_forest_optimized.pkl")
print("  • models/saved/scaler_optimized.pkl")

# Quick validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='f1')
print(f"\n📊 Validación cruzada rápida: F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
