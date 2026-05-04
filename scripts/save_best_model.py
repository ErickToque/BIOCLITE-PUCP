# =============================================================================
# scripts/save_best_model.py
# Guardar el mejor modelo Random Forest encontrado
# =============================================================================

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
from src.utils.helpers import set_seed

set_seed(42)

print("="*60)
print("💾 GUARDANDO MEJOR MODELO RANDOM FOREST")
print("="*60)

# Configuración
WINDOW_SIZE = 100
STEP_SIZE = 100

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

print(f"\n📊 Datos clínica ejercicio 6: {len(df_clinic):,} filas")

# Extraer features
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

print(f"Ventanas: {X.shape}")
print(f"Clases: Ausente={np.sum(y==0)}, Presente={np.sum(y==1)}")

# Escalar y balancear
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y==0)-1))
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Entrenar modelo final
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_balanced, y_balanced)

# Guardar modelo y scaler
joblib.dump(rf, 'models/saved/best_random_forest.pkl')
joblib.dump(scaler, 'models/saved/scaler.pkl')

print("\n✅ Modelo guardado en: models/saved/best_random_forest.pkl")
print("✅ Scaler guardado en: models/saved/scaler.pkl")

# Verificar
print(f"\n📊 Modelo info:")
print(f"  Features: {X.shape[1]}")
print(f"  Trees: {rf.n_estimators}")
print(f"  Max depth: {rf.max_depth}")