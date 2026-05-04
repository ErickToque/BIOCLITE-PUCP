# =============================================================================
# train_supervised_models.py - VERSIÓN CORREGIDA
# =============================================================================

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

from data_loader import BIOCLITEDataset
from preprocessing import IMUPreprocessor
from models import CNN1D, BiLSTM
from utils import set_seed, get_device, compute_metrics

set_seed(42)
device = get_device()
print(f"Device: {device}")

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
WINDOW_SIZE = 100
STEP_SIZE = 50

ejercicios = [4, 5, 6, 7, 8]
ejercicios_nombres = {
    4: "Pronación-supinación (manos)",
    5: "Tapping dedos",
    6: "Tapping pies (bradicinesia)",
    7: "Levantarse silla",
    8: "Marcha"
}

# =============================================================================
# 2. FUNCIÓN DE ENTRENAMIENTO CORREGIDA
# =============================================================================

def train_model(X_train, y_train, X_val, y_val, model_type='cnn', epochs=50, batch_size=32):
    """
    Entrena un modelo para clasificación binaria
    """
    # Convertir a tensores
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Crear data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Inicializar modelo
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    
    if model_type == 'cnn':
        model = CNN1D(input_channels=input_dim, seq_length=seq_len, num_classes=2)
    else:
        model = BiLSTM(input_size=input_dim, hidden_size=64, num_layers=2, num_classes=2)
    
    model = model.to(device)
    
    # Optimizador y loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
    
    # Si no hubo mejora, usar el modelo actual
    if best_model_state is None:
        best_model_state = model.state_dict().copy()
    
    # Cargar mejor modelo
    model.load_state_dict(best_model_state)
    
    return model

# =============================================================================
# 3. FUNCIÓN DE VALIDACIÓN CRUZADA CORREGIDA
# =============================================================================

def train_with_cross_validation(X, y, groups, exercise_num, model_type='cnn'):
    """
    Entrena usando Leave-One-Subject-Out cross validation
    """
    logo = LeaveOneGroupOut()
    
    results = {
        'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auc': []
    }
    
    fold_count = 0
    unique_groups = np.unique(groups)
    
    print(f"  Realizando LOSO con {len(unique_groups)} sujetos...")
    
    for train_idx, test_idx in logo.split(X, y, groups):
        # Verificar balance de clases en test
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Escalar
        scaler = RobustScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
        
        # Dividir train/val (80/20 dentro de train)
        n_train = int(len(X_train_scaled) * 0.8)
        X_tr, X_val = X_train_scaled[:n_train], X_train_scaled[n_train:]
        y_tr, y_val = y_train[:n_train], y_train[n_train:]
        
        # Verificar que val tenga ambas clases
        if len(np.unique(y_val)) < 2:
            # Usar parte de train como val
            X_tr = X_train_scaled[:-len(X_train_scaled)//5]
            X_val = X_train_scaled[-len(X_train_scaled)//5:]
            y_tr = y_train[:-len(y_train)//5]
            y_val = y_train[-len(y_train)//5:]
        
        # Entrenar
        model = train_model(X_tr, y_tr, X_val, y_val, model_type, epochs=30)
        
        # Evaluar
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_scaled).to(device)
            outputs = model(X_test_t)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
        
        metrics = compute_metrics(y_test, preds, probs)
        
        for k, v in metrics.items():
            results[k].append(v)
        
        fold_count += 1
        if fold_count % 5 == 0:
            print(f"    Procesados {fold_count}/{len(unique_groups)} folds...")
    
    return results, fold_count

# =============================================================================
# 4. MAIN: ENTRENAR POR EJERCICIO
# =============================================================================

print("\n" + "="*70)
print("🏋️ ENTRENAMIENTO DE MODELOS POR EJERCICIO (CONTEXTO SUPERVISADO)")
print("="*70)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Filtrar solo contexto supervisado (clínica inicial y final)
df_supervised = df[df['Contexto_sesion'].isin([1, 2])].copy()
print(f"\n📊 Datos supervisados: {len(df_supervised):,} filas")

preprocessor = IMUPreprocessor(fs=50)

all_results = {}

for ejercicio in ejercicios:
    print("\n" + "="*70)
    print(f"📋 EJERCICIO {ejercicio}: {ejercicios_nombres[ejercicio]}")
    print("="*70)
    
    # Filtrar ejercicio
    df_ej = df_supervised[df_supervised['Ejercicio'] == ejercicio].copy()
    
    if len(df_ej) == 0:
        print(f"  ⚠️ No hay datos para ejercicio {ejercicio}")
        continue
    
    print(f"  Muestras: {len(df_ej):,}")
    print(f"  Sesiones: {df_ej['Sesion'].nunique()}")
    print(f"  Sujetos: {df_ej['subject_id'].nunique()}")
    
    # Extraer ventanas y etiquetas
    X_windows, y_labels, groups = [], [], []
    
    for session in df_ej['Sesion'].unique():
        df_session = df_ej[df_ej['Sesion'] == session]
        
        # Extraer señales
        acc = df_session[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_session[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < WINDOW_SIZE:
            continue
        
        # Crear ventanas
        for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
            acc_window = acc[i:i+WINDOW_SIZE]
            gyro_window = gyro[i:i+WINDOW_SIZE]
            
            # Concatenar ACC y GYRO
            window = np.hstack([acc_window, gyro_window])
            X_windows.append(window)
            
            # Etiqueta basada en UPDRS (supervisado)
            updrs = df_session['UPDRS'].iloc[0]
            label = 1 if updrs not in [0, 99] else 0
            y_labels.append(label)
            
            groups.append(df_session['subject_id'].iloc[0])
    
    X = np.array(X_windows)
    y = np.array(y_labels)
    groups = np.array(groups)
    
    print(f"  Ventanas: {X.shape}")
    print(f"  Clases - Ausente: {np.sum(y==0)}, Presente: {np.sum(y==1)}")
    print(f"  Sujetos únicos: {len(np.unique(groups))}")
    
    if len(np.unique(y)) < 2:
        print(f"  ⚠️ Solo una clase presente, saltando...")
        continue
    
    # Entrenar con validación cruzada
    results, n_folds = train_with_cross_validation(X, y, groups, ejercicio, model_type='cnn')
    
    if len(results['accuracy']) == 0:
        print(f"  ⚠️ No se completaron folds válidos")
        continue
    
    all_results[ejercicio] = results
    
    # Mostrar resultados
    print(f"\n  📊 RESULTADOS ({n_folds} folds):")
    print(f"    Accuracy: {np.mean(results['accuracy']):.3f} ± {np.std(results['accuracy']):.3f}")
    print(f"    F1-score: {np.mean(results['f1']):.3f} ± {np.std(results['f1']):.3f}")
    print(f"    Recall:   {np.mean(results['recall']):.3f} ± {np.std(results['recall']):.3f}")
    print(f"    Precision:{np.mean(results['precision']):.3f} ± {np.std(results['precision']):.3f}")
    print(f"    AUC:      {np.mean(results['auc']):.3f} ± {np.std(results['auc']):.3f}")

# =============================================================================
# 5. GUARDAR RESULTADOS
# =============================================================================

print("\n" + "="*70)
print("💾 GUARDANDO RESULTADOS")
print("="*70)

# Guardar resultados en CSV
results_df = []
for ej, res in all_results.items():
    if len(res['accuracy']) > 0:
        results_df.append({
            'ejercicio': ej,
            'nombre': ejercicios_nombres[ej],
            'accuracy_mean': np.mean(res['accuracy']),
            'accuracy_std': np.std(res['accuracy']),
            'f1_mean': np.mean(res['f1']),
            'f1_std': np.std(res['f1']),
            'recall_mean': np.mean(res['recall']),
            'recall_std': np.std(res['recall']),
            'precision_mean': np.mean(res['precision']),
            'precision_std': np.std(res['precision']),
            'auc_mean': np.mean(res['auc']),
            'auc_std': np.std(res['auc']),
            'n_folds': len(res['accuracy'])
        })

if len(results_df) > 0:
    results_df = pd.DataFrame(results_df)
    results_df.to_csv('results_supervised_models.csv', index=False)
    print("✅ Resultados guardados en 'results_supervised_models.csv'")
    print("\n" + results_df.to_string())
else:
    print("⚠️ No se guardaron resultados")

print("\n🎉 Entrenamiento completado!")