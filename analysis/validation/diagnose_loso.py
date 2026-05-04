# analysis/diagnose_loso.py
import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*60)
print("🔍 DIAGNÓSTICO DE VALIDACIÓN LOSO")
print("="*60)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

# Extraer features
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
groups = np.array(groups)

print(f"\n📊 Dataset: {X.shape[0]} ventanas, {len(np.unique(groups))} sujetos")
print(f"Clases: {np.bincount(y)}")

# Analizar cada sujeto
logo = LeaveOneGroupOut()
valid_folds = 0
invalid_folds = []

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    test_subject = groups[test_idx][0]
    test_classes = np.unique(y[test_idx])
    
    if len(test_classes) < 2:
        invalid_folds.append({
            'subject': test_subject,
            'test_classes': test_classes.tolist(),
            'n_test_windows': len(test_idx)
        })
    else:
        valid_folds += 1

print(f"\n🔍 Folds válidos: {valid_folds}/{len(np.unique(groups))}")
print(f"Folds inválidos: {len(invalid_folds)}")

if invalid_folds:
    print("\n⚠️ Sujetos con una sola clase en test:")
    for inv in invalid_folds[:10]:
        print(f"  {inv['subject']}: {inv['test_classes']} ({inv['n_test_windows']} ventanas)")

# Solución: usar StratifiedGroupKFold
print("\n" + "="*60)
print("💡 SOLUCIÓN: StratifiedGroupKFold")
print("="*60)
print("""
En lugar de LOSO, usar StratifiedGroupKFold con n_splits=5 o 10.
Esto asegura que cada fold tenga ambas clases.
""")