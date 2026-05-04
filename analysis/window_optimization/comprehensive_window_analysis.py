# =============================================================================
# analysis/window_optimization/comprehensive_window_analysis.py
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 OPTIMIZACIÓN COMPREHENSIVA DE VENTANAS")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

print(f"\n📊 Datos: {len(df_clinic):,} filas")

# Configuraciones a probar
window_configs = {
    '0.5s': {'size': 25, 'step': 25},
    '1s': {'size': 50, 'step': 50},
    '2s': {'size': 100, 'step': 100},
    '3s': {'size': 150, 'step': 150},
    '4s': {'size': 200, 'step': 200},
    '5s': {'size': 250, 'step': 250},
    '6s': {'size': 300, 'step': 300},
    '8s': {'size': 400, 'step': 400},
    '10s': {'size': 500, 'step': 500},
}

preprocessor = IMUPreprocessor(fs=50)

def extract_windows(df_data, window_size, step_size):
    X, y, groups = [], [], []
    
    for session in df_data['Sesion'].unique():
        df_ses = df_data[df_data['Sesion'] == session]
        acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < window_size:
            continue
        
        for i in range(0, len(acc) - window_size, step_size):
            acc_window = acc[i:i+window_size]
            gyro_window = gyro[i:i+window_size]
            
            features = preprocessor.extract_features(acc_window, gyro_window)
            X.append(list(features.values()))
            
            updrs = df_ses['UPDRS'].iloc[0]
            label = 1 if updrs not in [0, 99] else 0
            y.append(label)
            groups.append(df_ses['subject_id'].iloc[0])
    
    return np.array(X), np.array(y), np.array(groups)

results = []

for name, config in tqdm(window_configs.items(), desc="Probando ventanas"):
    print(f"\n📊 Ventana {name} ({config['size']} samples)")
    
    X, y, groups = extract_windows(df_clinic, config['size'], config['step'])
    
    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"  ⚠️ Datos insuficientes")
        continue
    
    print(f"  Ventanas: {X.shape}, Clases: {np.bincount(y)}")
    
    # LOSO
    logo = LeaveOneGroupOut()
    f1_scores = []
    auc_scores = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Escalar
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest rápido
        '''# Modelo con overfitting
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        '''
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, 
                            min_samples_split=8, min_samples_leaf=4,
                            random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        y_pred = rf.predict(X_test_scaled)
        y_prob = rf.predict_proba(X_test_scaled)[:, 1]
        
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        auc_scores.append(roc_auc_score(y_test, y_prob))
    
    results.append({
        'window_name': name,
        'window_size': config['size'],
        'window_seconds': config['size'] / 50,
        'n_windows': X.shape[0],
        'n_features': X.shape[1],
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'n_folds': len(f1_scores)
    })
    
    print(f"  F1: {results[-1]['f1_mean']:.4f} ± {results[-1]['f1_std']:.4f}")
    print(f"  AUC: {results[-1]['auc_mean']:.4f} ± {results[-1]['auc_std']:.4f}")

# Resultados
results_df = pd.DataFrame(results)
results_df.to_csv('analysis/window_optimization/window_results.csv', index=False)

# Figura
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.errorbar(results_df['window_seconds'], results_df['f1_mean'], 
             yerr=results_df['f1_std'], fmt='o-', capsize=5, capthick=2)
ax1.set_xlabel('Tamaño de ventana (segundos)')
ax1.set_ylabel('F1-score')
ax1.set_title('F1-score vs Tamaño de ventana')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.errorbar(results_df['window_seconds'], results_df['auc_mean'], 
             yerr=results_df['auc_std'], fmt='s-', capsize=5, capthick=2, color='green')
ax2.set_xlabel('Tamaño de ventana (segundos)')
ax2.set_ylabel('AUC')
ax2.set_title('AUC vs Tamaño de ventana')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/window_optimization/window_optimization_results.png', dpi=300)
plt.close()

print("\n✅ Resultados guardados en analysis/window_optimization/")
print(results_df.to_string())