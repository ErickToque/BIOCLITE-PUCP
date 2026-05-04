# =============================================================================
# analysis/learning_curves.py
# Curvas de aprendizaje para demostrar que no hay overfitting
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*60)
print("📈 GENERANDO CURVAS DE APRENDIZAJE")
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

# Escalar
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Learning curve
train_sizes = np.linspace(0.1, 1.0, 10)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(n_estimators=150, max_depth=8, 
                            class_weight='balanced', random_state=42, n_jobs=-1)

train_sizes_abs, train_scores, test_scores = learning_curve(
    rf, X_scaled, y, 
    train_sizes=train_sizes, 
    cv=cv.split(X_scaled, y, groups),
    scoring='f1',
    n_jobs=-1,
    shuffle=True,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Figura
fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                alpha=0.1, color='blue')
ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, 
                alpha=0.1, color='red')

ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Entrenamiento')
ax.plot(train_sizes_abs, test_mean, 's-', color='red', label='Validación')

ax.axhline(y=0.789, color='green', linestyle='--', label='F1 final (0.789)')
ax.set_xlabel('Número de muestras de entrenamiento')
ax.set_ylabel('F1-score')
ax.set_title('Curva de Aprendizaje - Random Forest')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/statistical/learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Curva de aprendizaje guardada: analysis/statistical/learning_curve.png")

# Interpretación
gap = np.mean(train_mean[-3:] - test_mean[-3:])
print(f"\n📊 Brecha entrenamiento-validez: {gap:.4f}")
if gap < 0.05:
    print("  ✅ No hay overfitting significativo")
elif gap < 0.1:
    print("  🟡 Leve overfitting, aceptable")
else:
    print("  🔴 Posible overfitting, revisar modelo")