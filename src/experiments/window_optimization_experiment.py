# =============================================================================
# experiments/window_optimization_experiment.py
# Barrido de diferentes tamaños de ventana y estrategias
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score
from scipy.signal import resample
import warnings
import time
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.utils.helpers import set_seed, get_device

set_seed(42)
device = get_device()

# =============================================================================
# CONFIGURACIÓN DEL BARRIDO
# =============================================================================

# Diferentes tamaños de ventana (en segundos a 50Hz)
window_configs = {
    '1s': {'size': 50, 'step': 25},      # Ventana corta
    '2s': {'size': 100, 'step': 50},     # Ventana estándar
    '4s': {'size': 200, 'step': 100},    # Ventana larga
    '6s': {'size': 300, 'step': 150},    # Ventana muy larga
    '8s': {'size': 400, 'step': 200},    # Contexto amplio
}

# Diferentes estrategias de preprocesamiento
preprocessing_strategies = {
    'raw': 'Sin procesar',
    'filtered': 'Filtro pasa banda (0.5-20Hz)',
    'normalized': 'Normalización por ventana',
    'augmented': 'Aumentación (ruido + time warp)'
}

# =============================================================================
# ARQUITECTURAS A PROBAR (más livianas para barrido)
# =============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=6, seq_length=100, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(self.conv(x))


class DeepCNN(nn.Module):
    """CNN más profunda para ventanas largas"""
    def __init__(self, input_channels=6, seq_length=100, num_classes=2):
        super(DeepCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.classifier(self.conv(x))


class LSTMNet(nn.Module):
    """LSTM para secuencias largas"""
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# =============================================================================

from scipy import signal as sig

def apply_preprocessing(data, strategy, fs=50):
    """Aplica diferentes estrategias de preprocesamiento"""
    
    if strategy == 'filtered':
        # Filtro pasa banda para movimiento humano (0.5-20 Hz)
        nyquist = fs / 2
        b, a = sig.butter(4, [0.5/nyquist, 20/nyquist], btype='band')
        data = sig.filtfilt(b, a, data, axis=0)
        
    elif strategy == 'normalized':
        # Normalización por ventana
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-6
        data = (data - mean) / std
        
    elif strategy == 'augmented':
        # Aumentación: ruido gaussiano + time warp suave
        noise = np.random.normal(0, 0.05 * data.std(), data.shape)
        data = data + noise
        
        # Time warp (pequeña deformación temporal)
        if len(data) > 10:
            warp = np.linspace(0, 1, len(data))
            warp = np.sin(warp * np.pi) * 0.05 + warp
            idx = np.interp(np.linspace(0, 1, len(data)), warp, np.arange(len(data)))
            idx = np.clip(idx, 0, len(data)-1).astype(int)
            data = data[idx]
    
    return data

# =============================================================================
# EXTRACCIÓN DE VENTANAS CON DIFERENTES TAMAÑOS
# =============================================================================

def extract_windows(df_data, window_size, step_size, preprocessing='raw'):
    """Extrae ventanas con diferentes configuraciones"""
    
    X, y, groups = [], [], []
    
    for session in df_data['Sesion'].unique():
        df_ses = df_data[df_data['Sesion'] == session]
        acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < window_size:
            continue
        
        # Aplicar preprocesamiento a señal completa
        signal_full = np.hstack([acc, gyro])
        signal_full = apply_preprocessing(signal_full, preprocessing)
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

# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def train_and_evaluate(X, y, groups, model_class, epochs=50):
    """Entrena y evalúa un modelo con LOSO"""
    
    logo = LeaveOneGroupOut()
    results = {'f1': [], 'auc': []}
    
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Normalizar
        scaler = RobustScaler()
        X_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_flat).reshape(X_train.shape)
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
        
        # Train/val split
        val_size = int(len(X_train_scaled) * 0.2)
        X_tr, X_val = X_train_scaled[:-val_size], X_train_scaled[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        
        # Data loaders
        class_counts = np.bincount(y_tr)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[y_tr]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
            batch_size=32, sampler=sampler
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=32
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)),
            batch_size=32
        )
        
        # Modelo
        if model_class == 'LSTM':
            model = LSTMNet(input_size=X.shape[2], num_classes=2).to(device)
        elif model_class == 'DeepCNN':
            model = DeepCNN(input_channels=X.shape[2], seq_length=X.shape[1], num_classes=2).to(device)
        else:
            model = SimpleCNN(input_channels=X.shape[2], seq_length=X.shape[1], num_classes=2).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Entrenamiento rápido (20 épocas para barrido)
        for epoch in range(20):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluación
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(batch_y.numpy())
        
        results['f1'].append(f1_score(all_labels, all_preds, zero_division=0))
        results['auc'].append(roc_auc_score(all_labels, all_probs))
    
    return results

# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

print("="*80)
print("🔬 BARRIDO SISTEMÁTICO: TAMAÑO DE VENTANA + PREPROCESAMIENTO + ARQUITECTURA")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

print(f"\n📊 Datos clínica ejercicio 6: {len(df_clinic):,} filas")

# Barrido completo
results_summary = []

for window_name, config in window_configs.items():
    for strategy in preprocessing_strategies.keys():
        for model_name in ['SimpleCNN', 'DeepCNN', 'LSTM']:
            
            print(f"\n{'='*50}")
            print(f"Probando: Ventana={window_name}, Preproc={strategy}, Modelo={model_name}")
            print(f"{'='*50}")
            
            try:
                # Extraer ventanas
                X, y, groups = extract_windows(
                    df_clinic, 
                    window_size=config['size'],
                    step_size=config['step'],
                    preprocessing=strategy
                )
                
                print(f"  Ventanas: {X.shape}")
                print(f"  Clases: {np.bincount(y)}")
                
                if len(np.unique(y)) < 2 or X.shape[0] < 10:
                    print(f"  ⚠️ Datos insuficientes, saltando...")
                    continue
                
                # Entrenar y evaluar
                start_time = time.time()
                results = train_and_evaluate(X, y, groups, model_name, epochs=20)
                elapsed = time.time() - start_time
                
                if results['f1']:
                    results_summary.append({
                        'window': window_name,
                        'window_size': config['size'],
                        'preprocessing': strategy,
                        'model': model_name,
                        'n_windows': X.shape[0],
                        'f1_mean': np.mean(results['f1']),
                        'f1_std': np.std(results['f1']),
                        'auc_mean': np.mean(results['auc']),
                        'auc_std': np.std(results['auc']),
                        'time': elapsed
                    })
                    
                    print(f"  ✅ F1={results_summary[-1]['f1_mean']:.3f} ± {results_summary[-1]['f1_std']:.3f}")
                    print(f"     AUC={results_summary[-1]['auc_mean']:.3f} ± {results_summary[-1]['auc_std']:.3f}")
                else:
                    print(f"  ❌ Sin folds válidos")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")

# =============================================================================
# RESULTADOS FINALES
# =============================================================================

print("\n" + "="*80)
print("📊 RESULTADOS DEL BARRIDO SISTEMÁTICO")
print("="*80)

results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('f1_mean', ascending=False)

print(results_df.to_string())

# Guardar resultados
results_df.to_csv('window_optimization_results.csv', index=False)

# Mejor configuración encontrada
best = results_df.iloc[0]
print("\n" + "="*80)
print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
print("="*80)
print(f"  Ventana: {best['window']} ({best['window_size']} samples)")
print(f"  Preprocesamiento: {best['preprocessing']}")
print(f"  Modelo: {best['model']}")
print(f"  F1-score: {best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
print(f"  AUC: {best['auc_mean']:.3f} ± {best['auc_std']:.3f}")
print(f"  Ventanas: {best['n_windows']}")
print(f"  Tiempo: {best['time']:.1f}s")

# Comparar con Random Forest original
print("\n" + "="*80)
print("📊 COMPARACIÓN CON RANDOM FOREST (BASELINE)")
print("="*80)
print("Random Forest (ventana 2s, features manuales):")
print("  F1: 0.789 ± 0.016")
print("  AUC: 0.812 ± 0.049")
print(f"\nMejor DL encontrado:")
print(f"  F1: {best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
print(f"  AUC: {best['auc_mean']:.3f} ± {best['auc_std']:.3f}")