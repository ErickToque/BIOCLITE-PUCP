# =============================================================================
# deep_learning_models.py - VERSIÓN CORREGIDA
# =============================================================================

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from data_loader import BIOCLITEDataset
from utils import set_seed, get_device

set_seed(42)
device = get_device()
print(f"🔥 GPU disponible: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# 1. MODELOS AVANZADOS
# =============================================================================

class MultiScaleCNN(nn.Module):
    """CNN multiescala para capturar patrones temporales en diferentes resoluciones"""
    
    def __init__(self, input_channels=6, seq_length=100, num_classes=2):
        super(MultiScaleCNN, self).__init__()
        
        # Ramas con diferentes kernel sizes
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calcular dimensiones después de pooling
        self._to_linear = None
        
        # Capas compartidas después de concatenar
        self.shared = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        
        # Ajustar tamaños (pueden diferir por pooling)
        min_len = min(out1.shape[2], out2.shape[2], out3.shape[2])
        out1 = out1[:, :, :min_len]
        out2 = out2[:, :, :min_len]
        out3 = out3[:, :, :min_len]
        
        concat = torch.cat([out1, out2, out3], dim=1)
        return self.shared(concat)


class AttentionLSTM(nn.Module):
    """LSTM con atención temporal"""
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Atención temporal
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        return self.classifier(context)


class ResidualCNN(nn.Module):
    """CNN con conexiones residuales (ResNet1D)"""
    
    def __init__(self, input_channels=6, seq_length=100, num_classes=2):
        super(ResidualCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        # Bloques residuales
        self.res_block1 = self._make_res_block(64, 128)
        self.res_block2 = self._make_res_block(128, 256)
        self.res_block3 = self._make_res_block(256, 512)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        
        residual = x
        x = self.res_block1(x)
        # Ajustar dimensiones si es necesario
        if residual.shape[1] != x.shape[1]:
            residual = nn.functional.conv1d(residual, torch.ones(1, x.shape[1], 1).to(x.device), 
                                           groups=residual.shape[1])
        if residual.shape[2] != x.shape[2]:
            residual = nn.functional.interpolate(residual, size=x.shape[2])
        x = x + residual
        
        residual = x
        x = self.res_block2(x)
        if residual.shape[1] != x.shape[1]:
            residual = nn.functional.conv1d(residual, torch.ones(1, x.shape[1], 1).to(x.device),
                                           groups=residual.shape[1])
        if residual.shape[2] != x.shape[2]:
            residual = nn.functional.interpolate(residual, size=x.shape[2])
        x = x + residual
        
        residual = x
        x = self.res_block3(x)
        if residual.shape[1] != x.shape[1]:
            residual = nn.functional.conv1d(residual, torch.ones(1, x.shape[1], 1).to(x.device),
                                           groups=residual.shape[1])
        if residual.shape[2] != x.shape[2]:
            residual = nn.functional.interpolate(residual, size=x.shape[2])
        x = x + residual
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    """CNN simple y eficiente para comparación base"""
    
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
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.classifier(x)

# =============================================================================
# 2. FUNCIÓN DE ENTRENAMIENTO CORREGIDA
# =============================================================================

def train_model_gpu(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
    """Entrenamiento con early stopping y learning rate scheduling"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        scheduler.step(val_loss)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"    Early stopping en epoch {epoch}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_f1={val_f1:.4f}")
    
    # Si nunca hubo mejora, usar el modelo actual
    if best_model_state is None:
        best_model_state = model.state_dict().copy()
    
    model.load_state_dict(best_model_state)
    return model

# =============================================================================
# 3. EXPERIMENTO COMPLETO
# =============================================================================

print("\n" + "="*70)
print("🚀 COMPARACIÓN DE MODELOS DEEP LEARNING (GPU)")
print("="*70)

# Cargar datos (Ejercicio 6 - Bradicinesia)
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_supervised = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

print(f"\n📊 Datos supervisados ejercicio 6: {len(df_supervised):,} filas")

# Extraer ventanas (raw signals para DL)
WINDOW_SIZE = 100
STEP_SIZE = 100  # Sin solapamiento

X_raw = []
y_labels = []
groups = []

for session in df_supervised['Sesion'].unique():
    df_session = df_supervised[df_supervised['Sesion'] == session]
    
    acc = df_session[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyro = df_session[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
    
    if len(acc) < WINDOW_SIZE:
        continue
    
    for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
        window = np.hstack([acc[i:i+WINDOW_SIZE], gyro[i:i+WINDOW_SIZE]])
        X_raw.append(window)
        
        updrs = df_session['UPDRS'].iloc[0]
        label = 1 if updrs not in [0, 99] else 0
        y_labels.append(label)
        
        groups.append(df_session['subject_id'].iloc[0])

X_raw = np.array(X_raw)
y = np.array(y_labels)
groups = np.array(groups)

print(f"Ventanas: {X_raw.shape}")
print(f"Clases: Ausente={np.sum(y==0)}, Presente={np.sum(y==1)}")
print(f"Sujetos: {len(np.unique(groups))}")

# Normalizar
scaler = RobustScaler()
X_flat = X_raw.reshape(-1, X_raw.shape[-1])
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(X_raw.shape)

# Modelos a probar (más simples para evitar overfitting)
models_dict = {
    'SimpleCNN': SimpleCNN(input_channels=6, seq_length=WINDOW_SIZE),
    'MultiScaleCNN': MultiScaleCNN(input_channels=6, seq_length=WINDOW_SIZE),
    'AttentionLSTM': AttentionLSTM(input_size=6, hidden_size=64, num_layers=2),  # Reducido
}

# Validación cruzada
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
results = {name: {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auc': []} 
           for name in models_dict.keys()}

for name, model in models_dict.items():
    print(f"\n{'='*70}")
    print(f"📊 ENTRENANDO: {name}")
    print(f"{'='*70}")
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_scaled, y, groups)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Dividir train/val
        val_size = int(len(X_train) * 0.2)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        
        # Data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        # Balancear con sampler
        class_counts = np.bincount(y_tr)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[y_tr]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Crear nueva instancia del modelo para cada fold
        if name == 'SimpleCNN':
            model_fold = SimpleCNN(input_channels=6, seq_length=WINDOW_SIZE)
        elif name == 'MultiScaleCNN':
            model_fold = MultiScaleCNN(input_channels=6, seq_length=WINDOW_SIZE)
        else:
            model_fold = AttentionLSTM(input_size=6, hidden_size=64, num_layers=2)
        
        # Entrenar
        model_fold = train_model_gpu(model_fold, train_loader, val_loader, epochs=30)
        
        # Evaluar
        model_fold.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model_fold(batch_X)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(batch_y.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)
        
        results[name]['accuracy'].append(acc)
        results[name]['f1'].append(f1)
        results[name]['recall'].append(rec)
        results[name]['precision'].append(prec)
        results[name]['auc'].append(auc)
        
        fold_results.append(f1)
        print(f"  Fold {fold+1}: F1={f1:.3f}, AUC={auc:.3f}")
    
    print(f"  Media F1: {np.mean(fold_results):.3f} ± {np.std(fold_results):.3f}")

# Mostrar resultados comparativos
print("\n" + "="*70)
print("📊 RESULTADOS COMPARATIVOS")
print("="*70)

comparison = []
for name in results.keys():
    if len(results[name]['f1']) > 0:
        comparison.append({
            'Modelo': name,
            'Accuracy': f"{np.mean(results[name]['accuracy']):.3f} ± {np.std(results[name]['accuracy']):.3f}",
            'F1': f"{np.mean(results[name]['f1']):.3f} ± {np.std(results[name]['f1']):.3f}",
            'Recall': f"{np.mean(results[name]['recall']):.3f} ± {np.std(results[name]['recall']):.3f}",
            'Precision': f"{np.mean(results[name]['precision']):.3f} ± {np.std(results[name]['precision']):.3f}",
            'AUC': f"{np.mean(results[name]['auc']):.3f} ± {np.std(results[name]['auc']):.3f}"
        })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string())
comparison_df.to_csv('dl_models_comparison.csv', index=False)
print("\n✅ Resultados guardados en 'dl_models_comparison.csv'")

# Comparar con Random Forest anterior
print("\n" + "="*70)
print("📊 COMPARACIÓN CON RANDOM FOREST")
print("="*70)
print("Random Forest (resultado anterior):")
print("  F1: 0.789 ± 0.016")
print("  AUC: 0.812 ± 0.049")