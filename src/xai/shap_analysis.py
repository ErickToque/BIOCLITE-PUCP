# =============================================================================
# src/xai/shap_analysis.py
# Análisis de interpretabilidad con SHAP
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.preprocessing import RobustScaler

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor
from src.utils.helpers import set_seed

set_seed(42)

print("="*60)
print("🔬 ANÁLISIS SHAP - INTERPRETABILIDAD DEL MODELO")
print("="*60)

# Cargar modelo
print("\n📂 Cargando modelo...")
rf = joblib.load('models/saved/best_random_forest.pkl')
scaler = joblib.load('models/saved/scaler.pkl')
print("✅ Modelo cargado")

# Cargar datos
print("\n📊 Cargando datos...")
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

# Extraer features
WINDOW_SIZE = 100
STEP_SIZE = 100
preprocessor = IMUPreprocessor(fs=50)

X, y = [], []

print("\n🔧 Extrayendo features...")
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

X = np.array(X)
y = np.array(y)

print(f"✅ Datos: {X.shape[0]} ventanas, {X.shape[1]} features")

# Escalar
X_scaled = scaler.transform(X)

# Feature names
feature_names = list(preprocessor.extract_features(
    np.zeros((WINDOW_SIZE, 3)), 
    np.zeros((WINDOW_SIZE, 3))
).keys())

# Usar muestra más pequeña para SHAP
n_samples = min(300, len(X_scaled))
X_sample = X_scaled[:n_samples]
y_sample = y[:n_samples]

print(f"\n🔧 Creando explainer SHAP (esto puede tomar un minuto)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# SHAP para clase positiva
shap_values_pos = shap_values[:, :, 1]

print("✅ SHAP values calculados")

# =============================================================================
# FIGURAS
# =============================================================================

print("\n📈 Generando figuras...")

# 1. Summary plot (beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_pos, X_sample, feature_names=feature_names, 
                  show=False, max_display=15)
plt.title('SHAP Feature Importance - Top 15 Features', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ shap_beeswarm.png")

# 2. Bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_pos, X_sample, feature_names=feature_names, 
                  plot_type="bar", show=False, max_display=15)
plt.title('SHAP Mean |SHAP| - Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ shap_bar.png")

# 3. Tabla de importancia
mean_shap = np.abs(shap_values_pos).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_SHAP': mean_shap
}).sort_values('Mean_SHAP', ascending=False)

feature_importance_df.to_csv('results/tables/shap_feature_importance.csv', index=False)
print("  ✅ shap_feature_importance.csv")

print("\n📊 TOP 10 FEATURES:")
print(feature_importance_df.head(10).to_string())

print("\n✅ Análisis SHAP completado!")
