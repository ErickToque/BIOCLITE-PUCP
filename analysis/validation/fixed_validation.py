# =============================================================================
# analysis/fixed_validation.py - VERSIÓN CORREGIDA
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 VALIDACIÓN CORRECTA CON STRATIFIED GROUP K-FOLD")
print("="*80)

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

print(f"\n📊 Dataset: {X.shape[0]} ventanas, {X.shape[1]} features")
print(f"Clases: Ausente={np.sum(y==0)}, Presente={np.sum(y==1)}")
print(f"Sujetos: {len(np.unique(groups))}")

# =============================================================================
# Validación con StratifiedGroupKFold
# =============================================================================
print("\n" + "="*60)
print("📊 STRATIFIED GROUP K-FOLD VALIDATION")
print("="*60)

n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {
    'fold': [],
    'accuracy': [],
    'f1': [],
    'recall': [],
    'precision': [],
    'auc': []
}

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Escalar
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar
    '''# Modelo con overfitting
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    '''
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=8,
        min_samples_leaf=4, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    results['fold'].append(fold + 1)
    results['accuracy'].append(accuracy_score(y_test, y_pred))
    results['f1'].append(f1_score(y_test, y_pred))
    results['recall'].append(recall_score(y_test, y_pred))
    results['precision'].append(precision_score(y_test, y_pred))
    results['auc'].append(roc_auc_score(y_test, y_prob))
    
    print(f"Fold {fold+1}: F1={results['f1'][-1]:.4f}, AUC={results['auc'][-1]:.4f}")

# Resultados finales
print("\n" + "="*60)
print("📊 RESULTADOS FINALES")
print("="*60)
print(f"Accuracy:  {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
print(f"F1-score:  {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
print(f"Recall:    {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
print(f"Precision: {np.mean(results['precision']):.4f} ± {np.std(results['precision']):.4f}")
print(f"AUC:       {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")

# Guardar resultados
results_df = pd.DataFrame(results)
results_df.to_csv('analysis/fixed_validation_results.csv', index=False)
print("\n✅ Resultados guardados en analysis/fixed_validation_results.csv")