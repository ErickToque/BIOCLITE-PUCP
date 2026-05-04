# =============================================================================
# transfer_learning_final.py
# Transfer Learning: Clínica → Casa con Random Forest
# =============================================================================

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_loader import BIOCLITEDataset
from preprocessing import IMUPreprocessor
from utils import set_seed

set_seed(42)

# Configuración
WINDOW_SIZE = 100
STEP_SIZE = 100
THRESHOLD_75 = 0.75

print("="*70)
print("🔄 TRANSFER LEARNING: Clínica Supervisada → Casa No Supervisada")
print("="*70)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Ejercicio 6 (Bradicinesia)
ejercicio = 6
print(f"\n📋 Ejercicio {ejercicio}: Tapping pies (bradicinesia)")

# Separar por contexto
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == ejercicio)].copy()
df_home = df[(df['Contexto_sesion'] == 0) & (df['Ejercicio'] == ejercicio)].copy()

print(f"\n📊 Datos:")
print(f"  Clínica (supervisado): {len(df_clinic):,} muestras")
print(f"  Casa (no supervisado): {len(df_home):,} muestras")

# Extraer features
preprocessor = IMUPreprocessor(fs=50)

def extract_features_from_df(df_data):
    """Extrae features de un DataFrame"""
    X, y, subjects = [], [], []
    
    for session in df_data['Sesion'].unique():
        df_session = df_data[df_data['Sesion'] == session]
        
        acc = df_session[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_session[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < WINDOW_SIZE:
            continue
        
        for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
            acc_window = acc[i:i+WINDOW_SIZE]
            gyro_window = gyro[i:i+WINDOW_SIZE]
            
            features = preprocessor.extract_features(acc_window, gyro_window)
            X.append(list(features.values()))
            
            # Para clínica, usar UPDRS real; para casa, placeholder
            if 'UPDRS' in df_session.columns:
                updrs = df_session['UPDRS'].iloc[0]
                label = 1 if updrs not in [0, 99] else 0
                y.append(label)
            else:
                y.append(-1)  # Sin etiqueta
            
            subjects.append(df_session['subject_id'].iloc[0])
    
    return np.array(X), np.array(y), np.array(subjects)

print("\n🏥 Extrayendo features de clínica...")
X_clinic, y_clinic, subjects_clinic = extract_features_from_df(df_clinic)

print(f"  Ventanas: {X_clinic.shape}")
print(f"  Features: {X_clinic.shape[1]}")
print(f"  Clases: Ausente={np.sum(y_clinic==0)}, Presente={np.sum(y_clinic==1)}")

print("\n🏠 Extrayendo features de casa...")
X_home, _, subjects_home = extract_features_from_df(df_home)
print(f"  Ventanas: {X_home.shape}")

# =============================================================================
# ENTRENAR MODELO EN CLÍNICA
# =============================================================================
print("\n" + "="*70)
print("🎓 ENTRENANDO RANDOM FOREST EN DATOS DE CLÍNICA")
print("="*70)

# Escalar
scaler = RobustScaler()
X_clinic_scaled = scaler.fit_transform(X_clinic)

# Balancear con SMOTE
smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_clinic==0)-1))
X_balanced, y_balanced = smote.fit_resample(X_clinic_scaled, y_clinic)

# Random Forest optimizado
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_balanced, y_balanced)

# Validación en clínica
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X_clinic_scaled, y_clinic, cv=5, scoring='f1')
print(f"Validación cruzada (clínica): F1 = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# =============================================================================
# APLICAR A CASA
# =============================================================================
print("\n" + "="*70)
print("🏠 APLICANDO MODELO A DATOS DE CASA")
print("="*70)

X_home_scaled = scaler.transform(X_home)
y_home_proba = rf.predict_proba(X_home_scaled)[:, 1]
y_home_pred = (y_home_proba > 0.5).astype(int)

# Análisis por sujeto en casa
home_results = []
unique_subjects = np.unique(subjects_home)

for subject in unique_subjects:
    subject_mask = subjects_home == subject
    subject_probs = y_home_proba[subject_mask]
    subject_preds = y_home_pred[subject_mask]
    
    # Regla del 75% para determinar presencia del síntoma
    symptom_present = np.mean(subject_preds) >= THRESHOLD_75
    confidence = np.mean(subject_probs) if symptom_present else 1 - np.mean(subject_probs)
    
    home_results.append({
        'subject': subject,
        'n_windows': len(subject_probs),
        'mean_probability': np.mean(subject_probs),
        'std_probability': np.std(subject_probs),
        'symptom_present': symptom_present,
        'confidence': confidence,
        'n_positive_windows': np.sum(subject_preds),
        'positive_ratio': np.mean(subject_preds)
    })

home_df = pd.DataFrame(home_results)
home_df = home_df.sort_values('mean_probability', ascending=False)

print(f"\n📊 Predicciones por sujeto en casa:")
print(home_df.to_string())

# =============================================================================
# VISUALIZACIONES
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Distribución de probabilidades
axes[0, 0].hist(y_home_proba, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Umbral 0.5')
axes[0, 0].axvline(x=THRESHOLD_75, color='orange', linestyle='--', linewidth=2, label=f'Umbral 75%')
axes[0, 0].set_xlabel('Probabilidad de Bradicinesia')
axes[0, 0].set_ylabel('Número de ventanas')
axes[0, 0].set_title(f'Distribución en Casa (n={len(y_home_proba)})')
axes[0, 0].legend()

# 2. Por sujeto
subjects_sorted = home_df['subject'].values
probs_sorted = home_df['mean_probability'].values
colors = ['red' if p >= THRESHOLD_75 else 'green' for p in probs_sorted]

axes[0, 1].barh(range(len(subjects_sorted)), probs_sorted, color=colors, alpha=0.7)
axes[0, 1].axvline(x=THRESHOLD_75, color='black', linestyle='--', linewidth=2, label=f'Umbral {THRESHOLD_75}')
axes[0, 1].set_yticks(range(len(subjects_sorted)))
axes[0, 1].set_yticklabels(subjects_sorted, fontsize=8)
axes[0, 1].set_xlabel('Probabilidad promedio')
axes[0, 1].set_title('Predicción por Sujeto (Rojo≥75% = Bradicinesia)')
axes[0, 1].legend()

# 3. Feature importance
importances = rf.feature_importances_
feature_names = list(preprocessor.extract_features(np.zeros((100,3)), np.zeros((100,3))).keys())
top_idx = np.argsort(importances)[-15:]

axes[0, 2].barh(range(15), importances[top_idx])
axes[0, 2].set_yticks(range(15))
axes[0, 2].set_yticklabels([feature_names[i][:20] for i in top_idx])
axes[0, 2].set_xlabel('Importancia')
axes[0, 2].set_title('Top 15 Features para Bradicinesia')

# 4. Confianza por sujeto
conf_colors = ['darkgreen' if c > 0.7 else 'yellow' if c > 0.5 else 'red' for c in home_df['confidence'].values]
axes[1, 0].bar(range(len(home_df)), home_df['confidence'].values, color=conf_colors, alpha=0.7)
axes[1, 0].axhline(y=0.7, color='green', linestyle='--', label='Alta confianza')
axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', label='Confianza media')
axes[1, 0].set_xlabel('Sujeto')
axes[1, 0].set_ylabel('Confianza')
axes[1, 0].set_title('Confianza por Sujeto')
axes[1, 0].legend()

# 5. Ratio de ventanas positivas
axes[1, 1].bar(range(len(home_df)), home_df['positive_ratio'].values, alpha=0.7, color='purple')
axes[1, 1].axhline(y=THRESHOLD_75, color='red', linestyle='--', linewidth=2, label=f'Umbral {THRESHOLD_75}')
axes[1, 1].set_xlabel('Sujeto')
axes[1, 1].set_ylabel('Ratio ventanas positivas')
axes[1, 1].set_title(f'% de Ventanas con Bradicinesia (umbral: {THRESHOLD_75})')
axes[1, 1].legend()

# 6. Resumen texto
sospecha_count = home_df['symptom_present'].sum()
alta_confianza = (home_df['confidence'] > 0.7).sum()

axes[1, 2].axis('off')
axes[1, 2].text(0.05, 0.95, 
                f"📊 RESUMEN TRANSFER LEARNING\n"
                f"{'='*35}\n\n"
                f"🏥 Modelo entrenado en clínica:\n"
                f"   • F1-score: {cv_scores.mean():.3f}\n"
                f"   • Features: {X_clinic.shape[1]}\n\n"
                f"🏠 Predicciones en casa:\n"
                f"   • Sujetos evaluados: {len(home_df)}\n"
                f"   • Sospecha bradicinesia: {sospecha_count}\n"
                f"   • Alta confianza (>70%): {alta_confianza}\n\n"
                f"🎯 Sujetos con sospecha:\n"
                + "\n".join([f"   • {row['subject']} ({row['confidence']:.1%} confianza)" 
                            for _, row in home_df[home_df['symptom_present']].head(5).iterrows()]),
                fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.savefig('transfer_learning_final.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
home_df.to_csv('home_predictions_final.csv', index=False)

print("\n" + "="*70)
print("✅ TRANSFER LEARNING COMPLETADO")
print("="*70)
print(f"\n📁 Archivos guardados:")
print(f"   • home_predictions_final.csv - Predicciones por sujeto")
print(f"   • transfer_learning_final.png - Visualizaciones")
print(f"\n🎯 Resumen:")
print(f"   • {sospecha_count}/{len(home_df)} sujetos con sospecha de bradicinesia en casa")
print(f"   • Confianza promedio: {home_df['confidence'].mean():.2%}")