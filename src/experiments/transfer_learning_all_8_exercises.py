# =============================================================================
# transfer_learning_all_8_exercises.py
# Analizar TODOS los 8 ejercicios
# =============================================================================

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
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

# TODOS los 8 ejercicios
ejercicios = list(range(1, 9))
ejercicios_nombres = {
    1: "Habla",
    2: "Expresión facial", 
    3: "Temblor en reposo",
    4: "Pronación-supinación",
    5: "Tapping dedos",
    6: "Tapping pies (bradicinesia)",
    7: "Levantarse silla",
    8: "Marcha"
}

print("="*80)
print("🔄 TRANSFER LEARNING: ANÁLISIS COMPLETO (8 EJERCICIOS)")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

df_clinic = df[df['Contexto_sesion'].isin([1, 2])].copy()
df_home = df[df['Contexto_sesion'] == 0].copy()

print(f"\n📊 Datos totales:")
print(f"  Clínica: {len(df_clinic):,} muestras")
print(f"  Casa: {len(df_home):,} muestras")

preprocessor = IMUPreprocessor(fs=50)

def extract_features_from_df(df_data, ejercicio):
    """Extrae features de un DataFrame para un ejercicio específico"""
    df_ej = df_data[df_data['Ejercicio'] == ejercicio].copy()
    
    X = []
    y = []
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
            
            features = preprocessor.extract_features(acc_window, gyro_window)
            X.append(list(features.values()))
            
            if 'UPDRS' in df_session.columns:
                updrs = df_session['UPDRS'].iloc[0]
                label = 1 if updrs not in [0, 99] else 0
                y.append(label)
            else:
                y.append(-1)
            
            subjects.append(df_session['subject_id'].iloc[0])
    
    return np.array(X), np.array(y), np.array(subjects)

# Almacenar resultados
all_results = []

for ejercicio in ejercicios:
    print(f"\n{'='*60}")
    print(f"📋 EJERCICIO {ejercicio}: {ejercicios_nombres[ejercicio]}")
    print(f"{'='*60}")
    
    # Extraer datos de clínica
    X_clinic, y_clinic, subjects_clinic = extract_features_from_df(df_clinic, ejercicio)
    
    if len(X_clinic) == 0:
        print(f"  ⚠️ No hay datos suficientes")
        continue
    
    print(f"  Clínica: {X_clinic.shape[0]} ventanas, {X_clinic.shape[1]} features")
    print(f"    Clases: Ausente={np.sum(y_clinic==0)}, Presente={np.sum(y_clinic==1)}")
    
    if len(np.unique(y_clinic)) < 2:
        print(f"  ⚠️ Solo una clase, saltando...")
        continue
    
    # Extraer datos de casa
    X_home, _, subjects_home = extract_features_from_df(df_home, ejercicio)
    print(f"  Casa: {X_home.shape[0]} ventanas")
    
    # Entrenar Random Forest
    scaler = RobustScaler()
    X_clinic_scaled = scaler.fit_transform(X_clinic)
    
    smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_clinic==0)-1))
    X_balanced, y_balanced = smote.fit_resample(X_clinic_scaled, y_clinic)
    
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_balanced, y_balanced)
    
    # Evaluar en casa
    X_home_scaled = scaler.transform(X_home)
    probas = rf.predict_proba(X_home_scaled)[:, 1]
    
    # Análisis por sujeto con regla del 75%
    results = []
    for subject in np.unique(subjects_home):
        mask = subjects_home == subject
        subject_probs = probas[mask]
        subject_preds = (subject_probs > 0.5).astype(int)
        symptom_present = np.mean(subject_preds) >= THRESHOLD_75
        
        # Determinar grupo (0=sano, 1=Parkinson)
        grupo = int(subject.split('_')[0])
        
        results.append({
            'subject': subject,
            'grupo': grupo,
            'n_windows': len(subject_probs),
            'mean_probability': np.mean(subject_probs),
            'symptom_present': symptom_present,
            'confidence': np.mean(subject_probs) if symptom_present else 1 - np.mean(subject_probs)
        })
    
    df_results = pd.DataFrame(results)
    
    # Calcular métricas por grupo
    sanos = df_results[df_results['grupo'] == 0]
    parkinson = df_results[df_results['grupo'] == 1]
    
    sensibilidad = parkinson['symptom_present'].sum() / len(parkinson) if len(parkinson) > 0 else 0
    especificidad = 1 - (sanos['symptom_present'].sum() / len(sanos)) if len(sanos) > 0 else 0
    
    all_results.append({
        'ejercicio': ejercicio,
        'nombre': ejercicios_nombres[ejercicio],
        'n_ventanas_clinica': len(X_clinic),
        'n_ventanas_casa': len(X_home),
        'sensibilidad': sensibilidad,
        'especificidad': especificidad,
        'sospecha_parkinson': parkinson['symptom_present'].sum(),
        'total_parkinson': len(parkinson),
        'sospecha_sanos': sanos['symptom_present'].sum(),
        'total_sanos': len(sanos),
        'confianza_promedio': df_results['confidence'].mean()
    })
    
    print(f"\n  📊 RESULTADOS:")
    print(f"    Sanos: {sanos['symptom_present'].sum()}/{len(sanos)} positivos (espec={especificidad:.1%})")
    print(f"    Parkinson: {parkinson['symptom_present'].sum()}/{len(parkinson)} positivos (sens={sensibilidad:.1%})")
    print(f"    Confianza promedio: {df_results['confidence'].mean():.1%}")

# =============================================================================
# TABLA COMPARATIVA COMPLETA
# =============================================================================
print("\n" + "="*80)
print("📊 TABLA COMPARATIVA: TODOS LOS 8 EJERCICIOS")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('sensibilidad', ascending=False)

print(results_df[['ejercicio', 'nombre', 'sensibilidad', 'especificidad', 
                  'sospecha_parkinson', 'total_parkinson', 'confianza_promedio']].to_string())

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sensibilidad por ejercicio
ax1 = axes[0, 0]
colors = ['green' if s >= 0.7 else 'orange' if s >= 0.4 else 'red' for s in results_df['sensibilidad']]
ax1.bar(range(len(results_df)), results_df['sensibilidad'], color=colors)
ax1.axhline(y=0.7, color='green', linestyle='--', label='Bueno (70%)')
ax1.axhline(y=0.4, color='orange', linestyle='--', label='Aceptable (40%)')
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels([f"{r['ejercicio']}\n{r['nombre'][:10]}" for _, r in results_df.iterrows()], rotation=45, ha='right')
ax1.set_ylabel('Sensibilidad')
ax1.set_title('Sensibilidad por Ejercicio (Parkinson)')
ax1.legend()
ax1.set_ylim(0, 1)

# 2. Especificidad por ejercicio
ax2 = axes[0, 1]
colors_spec = ['green' if s >= 0.9 else 'orange' if s >= 0.7 else 'red' for s in results_df['especificidad']]
ax2.bar(range(len(results_df)), results_df['especificidad'], color=colors_spec)
ax2.axhline(y=0.9, color='green', linestyle='--', label='Excelente (90%)')
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels([f"{r['ejercicio']}\n{r['nombre'][:10]}" for _, r in results_df.iterrows()], rotation=45, ha='right')
ax2.set_ylabel('Especificidad')
ax2.set_title('Especificidad por Ejercicio (Sanos)')
ax2.legend()
ax2.set_ylim(0, 1)

# 3. Número de ventanas
ax3 = axes[1, 0]
ax3.bar(range(len(results_df)), results_df['n_ventanas_clinica'], alpha=0.7, label='Clínica')
ax3.bar(range(len(results_df)), results_df['n_ventanas_casa'], alpha=0.7, label='Casa', bottom=results_df['n_ventanas_clinica'])
ax3.set_xticks(range(len(results_df)))
ax3.set_xticklabels([f"{r['ejercicio']}\n{r['nombre'][:10]}" for _, r in results_df.iterrows()], rotation=45, ha='right')
ax3.set_ylabel('Número de ventanas')
ax3.set_title('Datos disponibles por ejercicio')
ax3.legend()

# 4. Tabla resumen
ax4 = axes[1, 1]
ax4.axis('off')
texto = "RESUMEN FINAL\n" + "="*30 + "\n\n"
for _, r in results_df.iterrows():
    status = "✅" if r['sensibilidad'] >= 0.7 else "🟡" if r['sensibilidad'] >= 0.4 else "❌"
    texto += f"{status} Ej{r['ejercicio']}: {r['nombre'][:15]} - Sens: {r['sensibilidad']:.0%}, Esp: {r['especificidad']:.0%}\n"
ax4.text(0.05, 0.95, texto, transform=ax4.transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.savefig('transfer_learning_all_8_exercises.png', dpi=300, bbox_inches='tight')
plt.show()

# Guardar resultados
results_df.to_csv('transfer_learning_8_exercises.csv', index=False)
print("\n✅ Resultados guardados en 'transfer_learning_8_exercises.csv'")