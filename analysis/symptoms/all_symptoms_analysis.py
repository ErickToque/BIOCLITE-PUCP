# =============================================================================
# analysis/all_symptoms_analysis.py
# Análisis completo para TODOS los síntomas (ejercicios 1-8)
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 ANÁLISIS COMPLETO PARA TODOS LOS SÍNTOMAS (Ejercicios 1-8)")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Configuración
WINDOW_SIZE = 100
STEP_SIZE = 100
preprocessor = IMUPreprocessor(fs=50)

# Ejercicios y sus síntomas
sintomas = {
    1: "Temblor postural",
    2: "Temblor en acción",
    3: "Temblor en reposo",
    4: "Pronación-supinación",
    5: "Tapping dedos",
    6: "Bradicinesia (tapping pies)",
    7: "Levantarse silla",
    8: "Marcha"
}

resultados = []

for ejercicio, nombre in sintomas.items():
    print(f"\n{'='*60}")
    print(f"📊 Analizando: Ejercicio {ejercicio} - {nombre}")
    print(f"{'='*60}")
    
    # Filtrar datos supervisados
    df_ej = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == ejercicio)].copy()
    
    if len(df_ej) == 0:
        print(f"  ⚠️ No hay datos para este ejercicio")
        continue
    
    # Extraer features y etiquetas
    X, y, groups = [], [], []
    
    for session in df_ej['Sesion'].unique():
        df_ses = df_ej[df_ej['Sesion'] == session]
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
    
    if len(X) == 0:
        print(f"  ⚠️ No hay ventanas suficientes")
        continue
    
    print(f"  Ventanas: {X.shape}")
    print(f"  Clases: Ausente={np.sum(y==0)}, Presente={np.sum(y==1)}")
    
    if len(np.unique(y)) < 2:
        print(f"  ⚠️ Solo una clase presente")
        continue
    
    # Validación cruzada
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    auc_scores = []
    
    for train_idx, test_idx in sgkf.split(X_scaled, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, 
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        f1_scores.append(f1_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))
    
    resultados.append({
        'ejercicio': ejercicio,
        'sintoma': nombre,
        'n_muestras': len(X),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'n_folds': len(f1_scores)
    })
    
    print(f"\n  📊 RESULTADOS:")
    print(f"    F1: {resultados[-1]['f1_mean']:.4f} ± {resultados[-1]['f1_std']:.4f}")
    print(f"    AUC: {resultados[-1]['auc_mean']:.4f} ± {resultados[-1]['auc_std']:.4f}")

# Tabla comparativa
resultados_df = pd.DataFrame(resultados)
print("\n" + "="*80)
print("📊 TABLA COMPARATIVA - TODOS LOS SÍNTOMAS")
print("="*80)
print(resultados_df.to_string())

# Guardar
resultados_df.to_csv('analysis/all_symptoms_results.csv', index=False)
print("\n✅ Resultados guardados en analysis/all_symptoms_results.csv")