# =============================================================================
# experiments/advanced_models_experiment.py
# Modelos robustos: LOSO, Attention, Skip Connections, Transformer
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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor
from src.utils.helpers import set_seed, get_device

set_seed(42)
device = get_device()
print(f"🔥 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

# =============================================================================
# ARQUITECTURAS AVANZADAS
# =============================================================================

class SkipConnectionCNN(nn.Module):
    """CNN con skip connections (ResNet1D)"""
    
    def __init__(self, input_channels=6, seq_length=100, num_classes=2):
        super(SkipConnectionCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        # Skip connection blocks
        self.res_block1 = self._make_res_block(64, 128)
        self.res_block2 = self._make_res_block(128, 256)
        self.res_block3 = self._make_res_block(256, 512)
        
        # Skip connection con proyección
        self.skip_proj1 = nn.Conv1d(64, 128, 1) if 64 != 128 else nn.Identity()
        self.skip_proj2 = nn.Conv1d(128, 256, 1) if 128 != 256 else nn.Identity()
        self.skip_proj3 = nn.Conv1d(256, 512, 1) if 256 != 512 else nn.Identity()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def _make_res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        
        # Skip connection 1
        residual = self.skip_proj1(x)
        x = self.res_block1(x)
        x = x + residual
        
        # Skip connection 2
        residual = self.skip_proj2(x)
        x = self.res_block2(x)
        x = x + residual
        
        # Skip connection 3
        residual = self.skip_proj3(x)
        x = self.res_block3(x)
        x = x + residual
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class MultiHeadAttentionLSTM(nn.Module):
    """LSTM con Multi-Head Self-Attention"""
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, 
                 num_heads=8, num_classes=2, dropout=0.3):
        super(MultiHeadAttentionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        return self.classifier(pooled)


class TCN(nn.Module):
    """Temporal Convolutional Network con dilations"""
    
    def __init__(self, input_size=6, num_channels=[64, 128, 256, 512], 
                 kernel_size=5, dropout=0.3, num_classes=2):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=dilation_size * (kernel_size - 1),
                    dilation=dilation_size
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class InceptionTime(nn.Module):
    """InceptionTime - Múltiples kernel sizes en paralelo"""
    
    def __init__(self, input_channels=6, num_classes=2):
        super(InceptionTime, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(input_channels, 32, kernel_size=11, padding=5)
        
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        
        self.conv5 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        
        concat = torch.cat([out1, out2, out3, out4], dim=1)
        concat = self.bn(concat)
        concat = self.relu(concat)
        
        out = self.conv5(concat)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.global_pool(out)
        out = out.squeeze(-1)
        return self.classifier(out)

# =============================================================================
# ENTRENAMIENTO CON LOSO
# =============================================================================

def train_model_loso(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """Entrenamiento con early stopping para LOSO"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
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
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

print("\n" + "="*80)
print("🚀 EXPERIMENTO AVANZADO: LOSO + ARQUITECTURAS PESADAS")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

# Extraer ventanas
WINDOW_SIZE = 100
STEP_SIZE = 50  # Con solapamiento para más datos

X_raw, y_labels, groups = [], [], []

for session in df_clinic['Sesion'].unique():
    df_ses = df_clinic[df_clinic['Sesion'] == session]
    acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
    
    if len(acc) < WINDOW_SIZE:
        continue
    
    for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
        window = np.hstack([acc[i:i+WINDOW_SIZE], gyro[i:i+WINDOW_SIZE]])
        X_raw.append(window)
        
        updrs = df_ses['UPDRS'].iloc[0]
        label = 1 if updrs not in [0, 99] else 0
        y_labels.append(label)
        groups.append(df_ses['subject_id'].iloc[0])

X_raw = np.array(X_raw)
y = np.array(y_labels)
groups = np.array(groups)

# Normalizar
scaler = RobustScaler()
X_flat = X_raw.reshape(-1, X_raw.shape[-1])
X_scaled = scaler.fit_transform(X_flat).reshape(X_raw.shape)

print(f"\n📊 Dataset: {X_scaled.shape[0]} ventanas, {X_scaled.shape[2]} canales")
print(f"   Clases: Ausente={np.sum(y==0)}, Presente={np.sum(y==1)}")
print(f"   Sujetos: {len(np.unique(groups))}")

# Modelos a probar
models_config = {
    'SkipConnectionCNN': SkipConnectionCNN(input_channels=6, seq_length=WINDOW_SIZE),
    'MultiHeadAttentionLSTM': MultiHeadAttentionLSTM(input_size=6, hidden_size=128, num_heads=8),
    'TCN': TCN(input_size=6, num_channels=[64, 128, 256, 512]),
    'InceptionTime': InceptionTime(input_channels=6),
}

# LOSO
logo = LeaveOneGroupOut()
results = {}

for name, model in models_config.items():
    print(f"\n{'='*60}")
    print(f"📊 ENTRENANDO: {name}")
    print(f"{'='*60}")
    
    fold_results = {'acc': [], 'f1': [], 'recall': [], 'precision': [], 'auc': []}
    fold_times = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        start_time = time.time()
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train/val split
        val_size = int(len(X_train) * 0.2)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        
        # Data loaders con balanceo
        class_counts = np.bincount(y_tr)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[y_tr]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
            batch_size=64, sampler=sampler
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=64
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=64
        )
        
        # Crear nueva instancia del modelo
        if name == 'SkipConnectionCNN':
            model_fold = SkipConnectionCNN(input_channels=6, seq_length=WINDOW_SIZE)
        elif name == 'MultiHeadAttentionLSTM':
            model_fold = MultiHeadAttentionLSTM(input_size=6, hidden_size=128, num_heads=8)
        elif name == 'TCN':
            model_fold = TCN(input_size=6, num_channels=[64, 128, 256, 512])
        else:
            model_fold = InceptionTime(input_channels=6)
        
        # Entrenar
        model_fold = train_model_loso(model_fold, train_loader, val_loader, epochs=100)
        
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
        
        fold_time = time.time() - start_time
        
        fold_results['acc'].append(accuracy_score(all_labels, all_preds))
        fold_results['f1'].append(f1_score(all_labels, all_preds, zero_division=0))
        fold_results['recall'].append(recall_score(all_labels, all_preds, zero_division=0))
        fold_results['precision'].append(precision_score(all_labels, all_preds, zero_division=0))
        fold_results['auc'].append(roc_auc_score(all_labels, all_probs))
        fold_times.append(fold_time)
        
        print(f"  Fold {fold+1}: F1={fold_results['f1'][-1]:.3f}, AUC={fold_results['auc'][-1]:.3f}, Time={fold_time:.1f}s")
    
    results[name] = fold_results
    
    print(f"\n  📊 RESULTADOS FINALES {name}:")
    print(f"    F1: {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
    print(f"    AUC: {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
    print(f"    Tiempo total: {np.sum(fold_times):.1f}s")

# Guardar resultados
results_df = pd.DataFrame([
    {
        'Modelo': name,
        'F1_mean': np.mean(res['f1']),
        'F1_std': np.std(res['f1']),
        'AUC_mean': np.mean(res['auc']),
        'AUC_std': np.std(res['auc']),
        'Accuracy_mean': np.mean(res['acc']),
        'Recall_mean': np.mean(res['recall']),
        'Precision_mean': np.mean(res['precision'])
    }
    for name, res in results.items()
])

results_df.to_csv('advanced_models_results.csv', index=False)
print("\n✅ Resultados guardados en 'advanced_models_results.csv'")
print(results_df.to_string())