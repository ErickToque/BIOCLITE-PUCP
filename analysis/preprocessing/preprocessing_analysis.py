import seaborn as sns
# =============================================================================
# analysis/preprocessing/preprocessing_analysis.py
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset

print("="*80)
print("🔬 ANÁLISIS DE PREPROCESAMIENTO")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

# Estrategias de filtrado
filter_configs = {
    'no_filter': None,
    'bandpass_0.5_20': (0.5, 20, 'bandpass'),
    'bandpass_0.5_10': (0.5, 10, 'bandpass'),
    'bandpass_1_8': (1, 8, 'bandpass'),
    'highpass_0.5': (0.5, None, 'highpass'),
    'lowpass_20': (None, 20, 'lowpass'),
}

# Estrategias de normalización
scaler_configs = {
    'robust': RobustScaler,
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'none': None
}

def apply_filter(data, config, fs=50):
    if config is None:
        return data
    
    nyquist = fs / 2
    if config[2] == 'bandpass':
        b, a = signal.butter(4, [config[0]/nyquist, config[1]/nyquist], btype='band')
    elif config[2] == 'highpass':
        b, a = signal.butter(4, config[0]/nyquist, btype='high')
    elif config[2] == 'lowpass':
        b, a = signal.butter(4, config[1]/nyquist, btype='low')
    
    return signal.filtfilt(b, a, data, axis=0)

# Extraer una ventana de ejemplo para visualización
session_example = df_clinic['Sesion'].iloc[0]
df_example = df_clinic[df_clinic['Sesion'] == session_example]
acc_example = df_example[['Acc_X', 'Acc_Y', 'Acc_Z']].values[:500]

# Visualizar efectos de filtros
fig, axes = plt.subplots(len(filter_configs), 1, figsize=(12, 12))

for idx, (filter_name, filter_config) in enumerate(filter_configs.items()):
    acc_filtered = apply_filter(acc_example.copy(), filter_config)
    acc_mag = np.sqrt(np.sum(acc_filtered**2, axis=1))
    
    axes[idx].plot(acc_mag, linewidth=0.8)
    axes[idx].set_ylabel('Aceleración (m/s²)')
    axes[idx].set_title(f'Filtro: {filter_name}')
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Muestra')
plt.tight_layout()
plt.savefig('analysis/preprocessing/filter_effects.png', dpi=300)
plt.close()

print("✅ Figura guardada: analysis/preprocessing/filter_effects.png")

# Ahora evaluar rendimiento con diferentes configuraciones
print("\n📊 Evaluando rendimiento...")

# Usar ventana fija de 2s sin solapamiento
WINDOW_SIZE = 100
STEP_SIZE = 100

from src.data.preprocessor import IMUPreprocessor
preprocessor = IMUPreprocessor(fs=50)

results = []

for filter_name, filter_config in filter_configs.items():
    for scaler_name, scaler_class in scaler_configs.items():
        print(f"\n  Probando: filter={filter_name}, scaler={scaler_name}")
        
        X, y, groups = [], [], []
        
        for session in df_clinic['Sesion'].unique():
            df_ses = df_clinic[df_clinic['Sesion'] == session]
            acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
            gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
            
            if len(acc) < WINDOW_SIZE:
                continue
            
            # Aplicar filtro
            signal_full = np.hstack([acc, gyro])
            signal_full = apply_filter(signal_full, filter_config)
            acc = signal_full[:, :3]
            gyro = signal_full[:, 3:]
            
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
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            continue
        
        # LOSO
        logo = LeaveOneGroupOut()
        f1_scores = []
        
        for train_idx, test_idx in logo.split(X, y, groups):
            if len(np.unique(y[test_idx])) < 2:
                continue
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Escalar
            if scaler_class is not None:
                scaler = scaler_class()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            '''# Modelo con overfitting
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            '''
            rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                            min_samples_split=8, min_samples_leaf=4,
                            random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
        results.append({
            'filter': filter_name,
            'scaler': scaler_name,
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'n_folds': len(f1_scores)
        })
        
        print(f"    F1: {results[-1]['f1_mean']:.4f} ± {results[-1]['f1_std']:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv('analysis/preprocessing/preprocessing_results.csv', index=False)

# Heatmap de resultados
pivot_table = results_df.pivot(index='filter', columns='scaler', values='f1_mean')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1)
plt.title('F1-score por combinación de filtro y normalización')
plt.tight_layout()
plt.savefig('analysis/preprocessing/preprocessing_heatmap.png', dpi=300)
plt.close()

print("\n✅ Resultados guardados en analysis/preprocessing/")
print(results_df.to_string())