# =============================================================================
# experiments/rigorous_validation.py
# Validación rigurosa de la configuración F1=1.0
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from scipy import signal
import warnings
import time
from collections import Counter
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.utils.helpers import set_seed, get_device

set_seed(42)
device = get_device()

print("="*80)
print("🔬 VALIDACIÓN RIGUROSA: ¿F1=1.0 REAL O OVERFITTING?")
print("="*80)

# =============================================================================
# CONFIGURACIONES A VALIDAR
# =============================================================================

configs_to_validate = [
    {
        'name': 'LSTM_8s_filtered',
        'window_size': 400,
        'step_size': 200,
        'preprocessing': 'filtered',
        'model_type': 'LSTM'
    },
    {
        'name': 'LSTM_6s_filtered',
        'window_size': 300,
        'step_size': 150,
        'preprocessing': 'filtered',
        'model_type': 'LSTM'
    },
    {
        'name': 'LSTM_1s_filtered',
        'window_size': 50,
        'step_size': 25,
        'preprocessing': 'filtered',
        'model_type': 'LSTM'
    },
    {
        'name': 'RandomForest_baseline',
        'window_size': 100,
        'step_size': 50,
        'preprocessing': 'features',
        'model_type': 'RF'
    }
]

# =============================================================================
# MODELO LSTM OPTIMIZADO
# =============================================================================

class OptimizedLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, 
                 dropout=0.3, num_classes=2):
        super(OptimizedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Usar el último hidden state de ambas direcciones
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        hidden_norm = self.layer_norm(hidden_concat)
        hidden_drop = self.dropout(hidden_norm)
        
        return self.classifier(hidden_drop)

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# =============================================================================

def apply_bandpass_filter(data, fs=50, lowcut=0.5, highcut=20, order=4):
    """Aplica filtro pasa banda"""
    nyquist = fs / 2
    b, a = signal.butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def extract_windows_with_filter(df_data, window_size, step_size, apply_filter=True):
    """Extrae ventanas con filtrado opcional"""
    X, y, groups = [], [], []
    
    for session in df_data['Sesion'].unique():
        df_ses = df_data[df_data['Sesion'] == session]
        acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < window_size:
            continue
        
        # Concatenar y filtrar
        signal_full = np.hstack([acc, gyro])
        
        if apply_filter:
            signal_full = apply_bandpass_filter(signal_full)
        
        acc = signal_full[:, :3]
        gyro = signal_full[:, 3:]
        
        for i in range(0, len(acc) - window_size, step_size):
            window = np.hstack([acc[i:i+window_size], gyro[i:i+window_size]])
            X.append(window)
            
            updrs = df_ses['UPDRS'].iloc[0]
            label = 1 if updrs not in [0, 99] else 0
            y.append(label)
            groups.append(df_ses['subject_id'].iloc[0])
    
    return np.array(X), np.array(y), np.array(groups)

def extract_features_for_rf(df_data, window_size, step_size):
    """Extrae features para Random Forest"""
    from src.data.preprocessor import IMUPreprocessor
    preprocessor = IMUPreprocessor(fs=50)
    
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

# =============================================================================
# VALIDACIÓN CRUZADA ESTRICTA
# =============================================================================

def train_lstm_fold(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100):
    """Entrena LSTM para un fold"""
    
    # Escalar
    scaler = RobustScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    # Balancear clases
    from imblearn.over_sampling import SMOTE
    X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    smote = SMOTE(random_state=42, k_neighbors=min(3, np.sum(y_train==0)-1))
    X_balanced, y_balanced = smote.fit_resample(X_train_flat, y_train)
    X_balanced = X_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    # Data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_balanced), torch.LongTensor(y_balanced))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Modelo
    model = OptimizedLSTM(input_size=X_train.shape[2], num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_y.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        scheduler.step(1 - val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 20:
            break
    
    # Cargar mejor modelo
    if best_state:
        model.load_state_dict(best_state)
    
    # Evaluación final
    model.eval()
    test_preds, test_probs, test_labels = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            test_preds.extend(preds)
            test_probs.extend(probs)
            test_labels.extend(batch_y.numpy())
    
    return {
        'f1': f1_score(test_labels, test_preds, zero_division=0),
        'auc': roc_auc_score(test_labels, test_probs),
        'acc': accuracy_score(test_labels, test_preds)
    }

def train_rf_fold(X_train, y_train, X_test, y_test):
    """Entrena Random Forest para un fold"""
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_balanced, y_balanced)
    
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob),
        'acc': accuracy_score(y_test, y_pred)
    }

# =============================================================================
# EJECUTAR VALIDACIÓN RIGUROSA
# =============================================================================

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

print(f"\n📊 Datos totales: {len(df_clinic):,} filas")

results = []

for config in configs_to_validate:
    print("\n" + "="*60)
    print(f"🔬 Validando: {config['name']}")
    print("="*60)
    
    # Extraer datos según configuración
    if config['model_type'] == 'RF':
        X, y, groups = extract_features_for_rf(
            df_clinic, 
            window_size=config['window_size'],
            step_size=config['step_size']
        )
    else:
        apply_filter = config['preprocessing'] == 'filtered'
        X, y, groups = extract_windows_with_filter(
            df_clinic,
            window_size=config['window_size'],
            step_size=config['step_size'],
            apply_filter=apply_filter
        )
    
    print(f"  Ventanas: {X.shape}")
    print(f"  Clases: {Counter(y)}")
    print(f"  Sujetos: {len(np.unique(groups))}")
    
    # Leave-One-Subject-Out Validation
    logo = LeaveOneGroupOut()
    fold_results = {'f1': [], 'auc': [], 'acc': []}
    fold_count = 0
    
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Split train/val
        val_size = int(len(X_train) * 0.2)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        
        if config['model_type'] == 'RF':
            res = train_rf_fold(X_tr, y_tr, X_test, y_test)
        else:
            res = train_lstm_fold(X_tr, y_tr, X_val, y_val, X_test, y_test, epochs=100)
        
        fold_results['f1'].append(res['f1'])
        fold_results['auc'].append(res['auc'])
        fold_results['acc'].append(res['acc'])
        fold_count += 1
        
        if fold_count % 5 == 0:
            print(f"    Fold {fold_count}: F1={res['f1']:.3f}, AUC={res['auc']:.3f}")
    
    # Estadísticas finales
    results.append({
        'config': config['name'],
        'window_size': config['window_size'],
        'preprocessing': config['preprocessing'],
        'model': config['model_type'],
        'n_windows': X.shape[0],
        'n_folds': fold_count,
        'f1_mean': np.mean(fold_results['f1']),
        'f1_std': np.std(fold_results['f1']),
        'auc_mean': np.mean(fold_results['auc']),
        'auc_std': np.std(fold_results['auc']),
        'acc_mean': np.mean(fold_results['acc']),
        'acc_std': np.std(fold_results['acc']),
        'f1_min': np.min(fold_results['f1']),
        'f1_max': np.max(fold_results['f1'])
    })
    
    print(f"\n  📊 RESULTADOS ({fold_count} folds):")
    print(f"    F1: {results[-1]['f1_mean']:.4f} ± {results[-1]['f1_std']:.4f}")
    print(f"    AUC: {results[-1]['auc_mean']:.4f} ± {results[-1]['auc_std']:.4f}")
    print(f"    Rango F1: [{results[-1]['f1_min']:.3f} - {results[-1]['f1_max']:.3f}]")

# =============================================================================
# TABLA COMPARATIVA FINAL
# =============================================================================
print("\n" + "="*80)
print("📊 TABLA COMPARATIVA: VALIDACIÓN RIGUROSA")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_mean', ascending=False)

print(results_df[['config', 'n_windows', 'f1_mean', 'f1_std', 'auc_mean', 
                  'auc_std', 'f1_min', 'f1_max']].to_string())

# Guardar resultados
results_df.to_csv('rigorous_validation_results.csv', index=False)

# =============================================================================
# CONCLUSIÓN
# =============================================================================
print("\n" + "="*80)
print("🎯 CONCLUSIÓN: ¿F1=1.0 REAL O OVERFITTING?")
print("="*80)

best = results_df.iloc[0]
if best['f1_std'] < 0.05 and best['f1_min'] > 0.9:
    print(f"✅ CONFIGURACIÓN ROBUSTA: {best['config']}")
    print(f"   F1={best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
    print(f"   El modelo generaliza bien")
elif best['f1_std'] > 0.1:
    print(f"⚠️ ALTA VARIABILIDAD: {best['config']}")
    print(f"   F1={best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
    print(f"   Posible overfitting - necesita más validación")
else:
    print(f"🟡 RENDIMIENTO MODERADO: {best['config']}")
    print(f"   F1={best['f1_mean']:.3f} ± {best['f1_std']:.3f}")

print("\n✅ Validación rigurosa completada")
print("📁 Resultados guardados en 'rigorous_validation_results.csv'")