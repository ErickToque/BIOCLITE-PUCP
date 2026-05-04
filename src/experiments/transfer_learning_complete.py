# =============================================================================
# transfer_learning_complete.py
# Transfer Learning: Clínica → Casa con TODOS los ejercicios y modelos
# =============================================================================

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from data_loader import BIOCLITEDataset
from preprocessing import IMUPreprocessor
from utils import set_seed, get_device

set_seed(42)
device = get_device()

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
WINDOW_SIZE = 100
STEP_SIZE = 100
THRESHOLD_75 = 0.75

ejercicios = [4, 5, 6, 7, 8]
ejercicios_nombres = {
    4: "Pronación-supinación",
    5: "Tapping dedos",
    6: "Tapping pies (bradicinesia)",
    7: "Levantarse silla",
    8: "Marcha"
}

# Modelos a probar
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
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.classifier(x)

# =============================================================================
# 2. FUNCIONES DE EXTRACCIÓN
# =============================================================================
preprocessor = IMUPreprocessor(fs=50)

def extract_features_and_raw(df_data, ejercicio):
    """Extrae tanto features como raw signals"""
    df_ej = df_data[df_data['Ejercicio'] == ejercicio].copy()
    
    X_features = []
    X_raw = []
    y_labels = []
    subjects = []
    
    for session in df_ej['Sesion'].unique():
        df_session = df_ej[df_ej['Sesion'] == session]
        
        acc = df_session[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_session[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < WINDOW_SIZE:
            continue
        
        for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
            acc_window = acc[i:i+WINDOW_SIZE]
            gyro_window = gyro[i:i+WINDOW_SIZE]
            
            # Features
            features = preprocessor.extract_features(acc_window, gyro_window)
            X_features.append(list(features.values()))
            
            # Raw signals
            raw_window = np.hstack([acc_window, gyro_window])
            X_raw.append(raw_window)
            
            # Label
            if 'UPDRS' in df_session.columns:
                updrs = df_session['UPDRS'].iloc[0]
                label = 1 if updrs not in [0, 99] else 0
                y_labels.append(label)
            else:
                y_labels.append(-1)
            
            subjects.append(df_session['subject_id'].iloc[0])
    
    return np.array(X_features), np.array(X_raw), np.array(y_labels), np.array(subjects)

# =============================================================================
# 3. ENTRENAMIENTO DE MODELOS
# =============================================================================
def train_random_forest(X_train, y_train):
    """Entrena Random Forest"""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train==0)-1))
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y_train)
    
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_balanced, y_balanced)
    
    return rf, scaler

def train_cnn(X_train, y_train):
    """Entrena CNN simple"""
    scaler = RobustScaler()
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(X_train.shape)
    
    # Balancear con SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(
        X_scaled.reshape(X_scaled.shape[0], -1), y_train
    )
    X_balanced = X_balanced.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    # Entrenar modelo
    model = SimpleCNN(input_channels=6, seq_length=WINDOW_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(torch.FloatTensor(X_balanced), torch.LongTensor(y_balanced))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(20):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    return model, scaler

# =============================================================================
# 4. EVALUACIÓN EN CASA
# =============================================================================
def evaluate_on_home(model, model_type, scaler, X_home_features, X_home_raw, subjects_home):
    """Evalúa modelo en datos de casa"""
    if model_type == 'rf':
        X_home_scaled = scaler.transform(X_home_features)
        probas = model.predict_proba(X_home_scaled)[:, 1]
    else:  # cnn
        X_flat = X_home_raw.reshape(-1, X_home_raw.shape[-1])
        X_scaled_flat = scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X_home_raw.shape)
        
        model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 64):
                batch = torch.FloatTensor(X_scaled[i:i+64]).to(device)
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                probas.extend(probs)
        probas = np.array(probas)
    
    # Análisis por sujeto con regla del 75%
    results = []
    for subject in np.unique(subjects_home):
        mask = subjects_home == subject
        subject_probs = probas[mask]
        subject_preds = (subject_probs > 0.5).astype(int)
        symptom_present = np.mean(subject_preds) >= THRESHOLD_75
        
        results.append({
            'subject': subject,
            'n_windows': len(subject_probs),
            'mean_probability': np.mean(subject_probs),
            'symptom_present': symptom_present,
            'confidence': np.mean(subject_probs) if symptom_present else 1 - np.mean(subject_probs)
        })
    
    return pd.DataFrame(results), probas

# =============================================================================
# 5. MAIN: TRANSFER LEARNING COMPLETO
# =============================================================================
print("="*80)
print("🔄 TRANSFER LEARNING COMPLETO: Clínica → Casa")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

df_clinic = df[df['Contexto_sesion'].isin([1, 2])].copy()
df_home = df[df['Contexto_sesion'] == 0].copy()

print(f"\n📊 Datos totales:")
print(f"  Clínica: {len(df_clinic):,} muestras")
print(f"  Casa: {len(df_home):,} muestras")

# Almacenar resultados
all_results = {}

for ejercicio in ejercicios:
    print("\n" + "="*80)
    print(f"📋 EJERCICIO {ejercicio}: {ejercicios_nombres[ejercicio]}")
    print("="*80)
    
    # Extraer datos
    print("Extrayendo features...")
    X_clinic_feat, X_clinic_raw, y_clinic, subjects_clinic = extract_features_and_raw(df_clinic, ejercicio)
    X_home_feat, X_home_raw, _, subjects_home = extract_features_and_raw(df_home, ejercicio)
    
    print(f"  Clínica: {X_clinic_feat.shape[0]} ventanas, {X_clinic_feat.shape[1]} features")
    print(f"  Casa: {X_home_feat.shape[0]} ventanas")
    
    if len(np.unique(y_clinic)) < 2:
        print(f"  ⚠️ Solo una clase en clínica, saltando...")
        continue
    
    # Entrenar Random Forest
    print("\n🎓 Entrenando Random Forest...")
    rf_model, rf_scaler = train_random_forest(X_clinic_feat, y_clinic)
    
    # Evaluar RF en casa
    print("🏠 Evaluando Random Forest en casa...")
    rf_results, rf_probas = evaluate_on_home(
        rf_model, 'rf', rf_scaler, X_home_feat, X_home_raw, subjects_home
    )
    
    # Entrenar CNN (solo si hay suficientes datos)
    print("\n🎓 Entrenando CNN...")
    try:
        cnn_model, cnn_scaler = train_cnn(X_clinic_raw, y_clinic)
        print("🏠 Evaluando CNN en casa...")
        cnn_results, cnn_probas = evaluate_on_home(
            cnn_model, 'cnn', cnn_scaler, X_home_feat, X_home_raw, subjects_home
        )
    except Exception as e:
        print(f"  ⚠️ CNN falló: {e}")
        cnn_results = None
    
    # Guardar resultados
    all_results[ejercicio] = {
        'rf': rf_results,
        'cnn': cnn_results,
        'rf_probas': rf_probas,
        'cnn_probas': cnn_probas
    }
    
    # Mostrar resumen
    print(f"\n📊 RESULTADOS TRANSFER LEARNING - Ejercicio {ejercicio}:")
    print(f"  Random Forest:")
    print(f"    Sujetos con sospecha: {rf_results['symptom_present'].sum()}/{len(rf_results)}")
    print(f"    Confianza promedio: {rf_results['confidence'].mean():.2%}")
    
    if cnn_results is not None:
        print(f"  CNN:")
        print(f"    Sujetos con sospecha: {cnn_results['symptom_present'].sum()}/{len(cnn_results)}")
        print(f"    Confianza promedio: {cnn_results['confidence'].mean():.2%}")
    
    # Ver concordancia entre modelos
    if cnn_results is not None:
        rf_pos = set(rf_results[rf_results['symptom_present']]['subject'])
        cnn_pos = set(cnn_results[cnn_results['symptom_present']]['subject'])
        agreement = len(rf_pos & cnn_pos) / len(rf_pos | cnn_pos) if len(rf_pos | cnn_pos) > 0 else 0
        print(f"  Concordancia RF-CNN: {agreement:.1%}")

# =============================================================================
# 6. VISUALIZACIÓN COMPARATIVA
# =============================================================================
print("\n" + "="*80)
print("📊 VISUALIZACIÓN COMPARATIVA")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (ej, results) in enumerate(all_results.items()):
    if idx >= 6:
        break
    
    rf_df = results['rf']
    cnn_df = results['cnn']
    
    # Ordenar por probabilidad
    rf_df = rf_df.sort_values('mean_probability', ascending=False)
    
    # Gráfico de barras RF
    colors_rf = ['red' if p >= THRESHOLD_75 else 'green' for p in rf_df['mean_probability'].values]
    axes[idx].barh(range(len(rf_df)), rf_df['mean_probability'].values, color=colors_rf, alpha=0.7)
    axes[idx].axvline(x=THRESHOLD_75, color='black', linestyle='--', linewidth=2)
    axes[idx].set_yticks(range(len(rf_df)))
    axes[idx].set_yticklabels(rf_df['subject'].values, fontsize=8)
    axes[idx].set_xlabel('Probabilidad promedio')
    axes[idx].set_title(f'Ej {ej}: {ejercicios_nombres[ej][:20]} (RF)')
    axes[idx].set_xlim(0, 1)

# Si hay más de 6 ejercicios, ajustar
for idx in range(len(all_results), 6):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('transfer_learning_all_exercises.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. TABLA COMPARATIVA FINAL
# =============================================================================
print("\n" + "="*80)
print("📊 TABLA COMPARATIVA: TODOS LOS EJERCICIOS")
print("="*80)

comparison_data = []
for ej, results in all_results.items():
    rf_df = results['rf']
    cnn_df = results['cnn']
    
    # Separar por grupo (asumiendo que subject 0_x son sanos, 1_x son Parkinson)
    rf_sanos = rf_df[rf_df['subject'].str.startswith('0_')]
    rf_parkinson = rf_df[rf_df['subject'].str.startswith('1_')]
    
    row = {
        'Ejercicio': ej,
        'Nombre': ejercicios_nombres[ej],
        'RF_Sanos_Positivos': f"{rf_sanos['symptom_present'].sum()}/{len(rf_sanos)}",
        'RF_Parkinson_Positivos': f"{rf_parkinson['symptom_present'].sum()}/{len(rf_parkinson)}",
        'RF_Sensibilidad': rf_parkinson['symptom_present'].sum() / len(rf_parkinson) if len(rf_parkinson) > 0 else 0,
        'RF_Especificidad': 1 - (rf_sanos['symptom_present'].sum() / len(rf_sanos)) if len(rf_sanos) > 0 else 0,
        'RF_Confianza': rf_df['confidence'].mean()
    }
    
    if cnn_df is not None:
        cnn_sanos = cnn_df[cnn_df['subject'].str.startswith('0_')]
        cnn_parkinson = cnn_df[cnn_df['subject'].str.startswith('1_')]
        row['CNN_Sanos_Positivos'] = f"{cnn_sanos['symptom_present'].sum()}/{len(cnn_sanos)}"
        row['CNN_Parkinson_Positivos'] = f"{cnn_parkinson['symptom_present'].sum()}/{len(cnn_parkinson)}"
        row['CNN_Sensibilidad'] = cnn_parkinson['symptom_present'].sum() / len(cnn_parkinson) if len(cnn_parkinson) > 0 else 0
        row['CNN_Especificidad'] = 1 - (cnn_sanos['symptom_present'].sum() / len(cnn_sanos)) if len(cnn_sanos) > 0 else 0
    
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string())
comparison_df.to_csv('transfer_learning_complete_results.csv', index=False)

print("\n" + "="*80)
print("✅ TRANSFER LEARNING COMPLETO FINALIZADO")
print("="*80)
print("\n📁 Archivos generados:")
print("  • transfer_learning_all_exercises.png - Visualizaciones")
print("  • transfer_learning_complete_results.csv - Resultados comparativos")