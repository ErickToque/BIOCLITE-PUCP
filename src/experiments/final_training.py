# =============================================================================
# src/experiments/final_training.py
# Script unificado para entrenamiento final del modelo
# =============================================================================

"""
Script de entrenamiento final para el modelo de bradicinesia.

Este script:
1. Carga los datos del ejercicio 6 (tapping pies)
2. Entrena un Random Forest con validación cruzada
3. Evalúa el rendimiento en contexto supervisado y no supervisado
4. Guarda el modelo final para producción
"""

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor
from src.utils.helpers import set_seed, get_device, compute_metrics

set_seed(42)

print("="*80)
print("🏆 ENTRENAMIENTO FINAL - MODELO DE BRADICINESIA")
print("="*80)

# Configuración
WINDOW_SIZE = 100
STEP_SIZE = 100
THRESHOLD_75 = 0.75

# Cargar datos
print("\n📊 Cargando datos...")
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Usar ejercicio 6 (tapping pies) en contexto supervisado
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()
df_home = df[(df['Contexto_sesion'] == 0) & (df['Ejercicio'] == 6)].copy()

print(f"  Clínica: {len(df_clinic):,} filas")
print(f"  Casa: {len(df_home):,} filas")

# Extraer features
print("\n🔧 Extrayendo features...")
preprocessor = IMUPreprocessor(fs=50)

def extract_features(df_data):
    X, y, subjects = [], [], []
    
    for session in df_data['Sesion'].unique():
        df_ses = df_data[df_data['Sesion'] == session]
        acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < WINDOW_SIZE:
            continue
        
        for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
            acc_window = acc[i:i+WINDOW_SIZE]
            gyro_window = gyro[i:i+WINDOW_SIZE]
            
            features = preprocessor.extract_features(acc_window, gyro_window)
            X.append(list(features.values()))
            
            if 'UPDRS' in df_ses.columns:
                updrs = df_ses['UPDRS'].iloc[0]
                label = 1 if updrs not in [0, 99] else 0
                y.append(label)
            else:
                y.append(-1)
            
            subjects.append(df_ses['subject_id'].iloc[0])
    
    return np.array(X), np.array(y), np.array(subjects)

X_clinic, y_clinic, subjects_clinic = extract_features(df_clinic)
X_home, _, subjects_home = extract_features(df_home)

print(f"  Clínica: {X_clinic.shape[0]} ventanas, {X_clinic.shape[1]} features")
print(f"  Casa: {X_home.shape[0]} ventanas")

# Escalar y balancear
print("\n⚖️ Balanceando datos...")
scaler = RobustScaler()
X_clinic_scaled = scaler.fit_transform(X_clinic)

smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_clinic==0)-1))
X_balanced, y_balanced = smote.fit_resample(X_clinic_scaled, y_clinic)

print(f"  Balanceado: {X_balanced.shape[0]} muestras")

# Entrenar modelo final
print("\n🎓 Entrenando Random Forest...")
'''
# Modelo con overfitting
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
'''
rf = RandomForestClassifier(
    n_estimators=150,          # REDUCIDO: 300 → 150
    max_depth=8,               # REDUCIDO: 12 → 8
    min_samples_split=8,       # AUMENTADO: 5 → 8
    min_samples_leaf=4,        # AUMENTADO: 2 → 4
    max_features='sqrt',       # NUEVO: limitar features por árbol
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_balanced, y_balanced)

# Evaluación en clínica
print("\n📊 Evaluación en clínica (cross-validation)...")
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X_clinic_scaled, y_clinic, cv=5, scoring='f1')
print(f"  F1 promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Transfer learning a casa
print("\n🏠 Aplicando a datos de casa...")
X_home_scaled = scaler.transform(X_home)
y_home_proba = rf.predict_proba(X_home_scaled)[:, 1]

# Análisis por sujeto
home_results = []
for subject in np.unique(subjects_home):
    mask = subjects_home == subject
    subject_probs = y_home_proba[mask]
    subject_preds = (subject_probs > 0.5).astype(int)
    
    grupo = int(subject.split('_')[0])
    symptom_present = np.mean(subject_preds) >= THRESHOLD_75
    
    home_results.append({
        'subject': subject,
        'grupo': 'Parkinson' if grupo == 1 else 'Sano',
        'n_windows': len(subject_probs),
        'probability': np.mean(subject_probs),
        'symptom_present': symptom_present,
        'confidence': np.mean(subject_probs) if symptom_present else 1 - np.mean(subject_probs)
    })

home_df = pd.DataFrame(home_results)
print(f"\n📊 Resultados en casa:")
print(f"  Sanos: {home_df[home_df['grupo']=='Sano']['symptom_present'].sum()}/{len(home_df[home_df['grupo']=='Sano'])} positivos")
print(f"  Parkinson: {home_df[home_df['grupo']=='Parkinson']['symptom_present'].sum()}/{len(home_df[home_df['grupo']=='Parkinson'])} positivos")

# Guardar modelo
print("\n💾 Guardando modelo...")
joblib.dump(rf, 'models/saved/best_random_forest.pkl')
joblib.dump(scaler, 'models/saved/scaler.pkl')
print("  ✅ Modelo guardado en models/saved/")

# Guardar resultados
home_df.to_csv('results/tables/home_predictions_final.csv', index=False)
print("  ✅ Resultados guardados en results/tables/")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO FINAL COMPLETADO")
print("="*80)
print(f"""
📊 RESUMEN FINAL:
  Modelo: Random Forest
  Ejercicio: 6 (Tapping pies)
  Features: {X_clinic.shape[1]}
  F1 (clínica): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  Sensibilidad (casa): {home_df[home_df['grupo']=='Parkinson']['symptom_present'].sum() / len(home_df[home_df['grupo']=='Parkinson']):.1%}
  Especificidad (casa): {1 - home_df[home_df['grupo']=='Sano']['symptom_present'].sum() / len(home_df[home_df['grupo']=='Sano']):.1%}
""")