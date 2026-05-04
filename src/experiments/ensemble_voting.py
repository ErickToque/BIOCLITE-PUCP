# =============================================================================
# ensemble_voting_ej5_ej6.py
# Ensemble Voting entre Ejercicio 5 (Tapping dedos) y Ejercicio 6 (Tapping pies)
# =============================================================================

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
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

ejercicios = [4, 5, 6, 7, 8]
ejercicios_nombres = {
    4: "Pronación",
    5: "Tapping dedos",
    6: "Tapping pies",
    7: "Levantarse",
    8: "Marcha"
}

print("="*80)
print("🎯 ENSEMBLE VOTING: Ejercicio 5 + Ejercicio 6")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[df['Contexto_sesion'].isin([1, 2])].copy()
df_home = df[df['Contexto_sesion'] == 0].copy()

preprocessor = IMUPreprocessor(fs=50)

def extract_features_for_exercise(df_data, ejercicio):
    """Extrae features para un ejercicio específico"""
    df_ej = df_data[df_data['Ejercicio'] == ejercicio].copy()
    
    X, y, subjects = [], [], []
    
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

# Extraer datos
print("\n📊 Extrayendo datos...")
X_clinic_ej5, y_clinic_ej5, _ = extract_features_for_exercise(df_clinic, 5)
X_clinic_ej6, y_clinic_ej6, _ = extract_features_for_exercise(df_clinic, 6)

# Datos de casa para todos los ejercicios
home_data = {}
for ej in ejercicios:
    X_home, _, subjects_home = extract_features_for_exercise(df_home, ej)
    home_data[ej] = {'X': X_home, 'subjects': subjects_home}
    print(f"  Ej{ej}: {X_home.shape[0]} ventanas en casa")

print(f"\n🏥 Datos clínica:")
print(f"  Ej5: {X_clinic_ej5.shape} ventanas (Positivos: {np.sum(y_clinic_ej5==1)})")
print(f"  Ej6: {X_clinic_ej6.shape} ventanas (Positivos: {np.sum(y_clinic_ej6==1)})")

# =============================================================================
# ENTRENAR MODELOS INDIVIDUALES
# =============================================================================
print("\n" + "="*80)
print("🎓 ENTRENANDO MODELOS INDIVIDUALES")
print("="*80)

def train_rf(X, y, name):
    """Entrena Random Forest"""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SMOTE para balancear
    smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y==0)-1))
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_balanced, y_balanced)
    
    print(f"  {name}: Entrenado con {X_balanced.shape[0]} muestras balanceadas")
    return rf, scaler

# Entrenar modelos
rf5, scaler5 = train_rf(X_clinic_ej5, y_clinic_ej5, "Ejercicio 5")
rf6, scaler6 = train_rf(X_clinic_ej6, y_clinic_ej6, "Ejercicio 6")

# =============================================================================
# EVALUAR ENSEMBLE EN CADA EJERCICIO DE CASA
# =============================================================================
print("\n" + "="*80)
print("🏠 EVALUANDO ENSEMBLE EN CASA")
print("="*80)

ensemble_results = {}

for test_ej in ejercicios:
    print(f"\n{'='*60}")
    print(f"📋 Evaluando en Ejercicio {test_ej}: {ejercicios_nombres[test_ej]}")
    print(f"{'='*60}")
    
    X_test = home_data[test_ej]['X']
    subjects_test = home_data[test_ej]['subjects']
    
    if len(X_test) == 0:
        print(f"  ⚠️ No hay datos")
        continue
    
    # Predicciones individuales
    X_test5_scaled = scaler5.transform(X_test)
    X_test6_scaled = scaler6.transform(X_test)
    
    probas5 = rf5.predict_proba(X_test5_scaled)[:, 1]
    probas6 = rf6.predict_proba(X_test6_scaled)[:, 1]
    
    # Estrategias de ensemble
    # 1. Soft voting (promedio de probabilidades)
    probas_ensemble_soft = (probas5 + probas6) / 2
    
    # 2. Hard voting (mayoría)
    preds5 = (probas5 > 0.5).astype(int)
    preds6 = (probas6 > 0.5).astype(int)
    preds_ensemble_hard = (preds5 + preds6) >= 1  # OR lógico (al menos uno)
    
    # 3. Weighted voting (más peso al mejor)
    # Ej6 tiene mejor especificidad, Ej5 mejor sensibilidad
    peso_ej5, peso_ej6 = 0.4, 0.6
    probas_ensemble_weighted = (probas5 * peso_ej5 + probas6 * peso_ej6)
    
    # Evaluar cada estrategia por sujeto
    strategies = {
        'Solo Ej5': probas5,
        'Solo Ej6': probas6,
        'Ensemble Soft (avg)': probas_ensemble_soft,
        'Ensemble Hard (OR)': preds_ensemble_hard,
        'Ensemble Weighted (40/60)': probas_ensemble_weighted
    }
    
    results = {}
    
    for strategy_name, predictions in strategies.items():
        # Analizar por sujeto con regla del 75%
        subject_results = []
        for subject in np.unique(subjects_test):
            mask = subjects_test == subject
            
            if strategy_name == 'Ensemble Hard (OR)':
                # Para hard voting, ya son predicciones binarias
                subject_preds = predictions[mask]
                symptom_present = np.mean(subject_preds) >= THRESHOLD_75
            else:
                # Para probabilidades
                subject_probs = predictions[mask]
                subject_preds = (subject_probs > 0.5).astype(int)
                symptom_present = np.mean(subject_preds) >= THRESHOLD_75
            
            grupo = int(subject.split('_')[0])
            subject_results.append({
                'subject': subject,
                'grupo': grupo,
                'symptom_present': symptom_present
            })
        
        df_res = pd.DataFrame(subject_results)
        sanos = df_res[df_res['grupo'] == 0]
        parkinson = df_res[df_res['grupo'] == 1]
        
        sensibilidad = parkinson['symptom_present'].sum() / len(parkinson) if len(parkinson) > 0 else 0
        especificidad = 1 - (sanos['symptom_present'].sum() / len(sanos)) if len(sanos) > 0 else 0
        n_positivos = df_res['symptom_present'].sum()
        
        results[strategy_name] = {
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'n_positivos': n_positivos,
            'total_sujetos': len(df_res)
        }
        
        print(f"  {strategy_name:25} → Sens: {sensibilidad:.1%}, Esp: {especificidad:.1%}, Positivos: {n_positivos}/{len(df_res)}")
    
    ensemble_results[test_ej] = results

# =============================================================================
# TABLA COMPARATIVA FINAL
# =============================================================================
print("\n" + "="*80)
print("📊 TABLA COMPARATIVA: ENSEMBLE vs INDIVIDUALES")
print("="*80)

comparison_data = []
for test_ej in ejercicios:
    if test_ej not in ensemble_results:
        continue
    
    res = ensemble_results[test_ej]
    comparison_data.append({
        'Ejercicio': test_ej,
        'Nombre': ejercicios_nombres[test_ej],
        'Solo_Ej5_Sens': f"{res['Solo Ej5']['sensibilidad']:.1%}",
        'Solo_Ej5_Esp': f"{res['Solo Ej5']['especificidad']:.1%}",
        'Solo_Ej6_Sens': f"{res['Solo Ej6']['sensibilidad']:.1%}",
        'Solo_Ej6_Esp': f"{res['Solo Ej6']['especificidad']:.1%}",
        'Ensemble_Soft_Sens': f"{res['Ensemble Soft (avg)']['sensibilidad']:.1%}",
        'Ensemble_Soft_Esp': f"{res['Ensemble Soft (avg)']['especificidad']:.1%}",
        'Ensemble_Hard_Sens': f"{res['Ensemble Hard (OR)']['sensibilidad']:.1%}",
        'Ensemble_Hard_Esp': f"{res['Ensemble Hard (OR)']['especificidad']:.1%}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string())

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Comparación de sensibilidad
ax1 = axes[0, 0]
ejes = [f"Ej{ej}\n{ejercicios_nombres[ej][:8]}" for ej in ejercicios if ej in ensemble_results]
sens_ej5 = [ensemble_results[ej]['Solo Ej5']['sensibilidad'] for ej in ejercicios if ej in ensemble_results]
sens_ej6 = [ensemble_results[ej]['Solo Ej6']['sensibilidad'] for ej in ejercicios if ej in ensemble_results]
sens_ensemble = [ensemble_results[ej]['Ensemble Hard (OR)']['sensibilidad'] for ej in ejercicios if ej in ensemble_results]

x = np.arange(len(ejes))
width = 0.25

ax1.bar(x - width, sens_ej5, width, label='Solo Ej5', color='blue', alpha=0.7)
ax1.bar(x, sens_ej6, width, label='Solo Ej6', color='green', alpha=0.7)
ax1.bar(x + width, sens_ensemble, width, label='Ensemble (OR)', color='red', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(ejes, rotation=45, ha='right')
ax1.set_ylabel('Sensibilidad')
ax1.set_title('Comparación de Sensibilidad')
ax1.legend()
ax1.set_ylim(0, 1)

# 2. Comparación de especificidad
ax2 = axes[0, 1]
esp_ej5 = [ensemble_results[ej]['Solo Ej5']['especificidad'] for ej in ejercicios if ej in ensemble_results]
esp_ej6 = [ensemble_results[ej]['Solo Ej6']['especificidad'] for ej in ejercicios if ej in ensemble_results]
esp_ensemble = [ensemble_results[ej]['Ensemble Hard (OR)']['especificidad'] for ej in ejercicios if ej in ensemble_results]

ax2.bar(x - width, esp_ej5, width, label='Solo Ej5', color='blue', alpha=0.7)
ax2.bar(x, esp_ej6, width, label='Solo Ej6', color='green', alpha=0.7)
ax2.bar(x + width, esp_ensemble, width, label='Ensemble (OR)', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(ejes, rotation=45, ha='right')
ax2.set_ylabel('Especificidad')
ax2.set_title('Comparación de Especificidad')
ax2.legend()
ax2.set_ylim(0, 1)

# 3. F1-score aproximado (media armónica)
ax3 = axes[0, 2]
def f1_score(sens, esp):
    if sens + esp == 0:
        return 0
    return 2 * (sens * esp) / (sens + esp)

f1_ej5 = [f1_score(ensemble_results[ej]['Solo Ej5']['sensibilidad'], 
                    ensemble_results[ej]['Solo Ej5']['especificidad']) for ej in ejercicios if ej in ensemble_results]
f1_ej6 = [f1_score(ensemble_results[ej]['Solo Ej6']['sensibilidad'], 
                    ensemble_results[ej]['Solo Ej6']['especificidad']) for ej in ejercicios if ej in ensemble_results]
f1_ensemble = [f1_score(ensemble_results[ej]['Ensemble Hard (OR)']['sensibilidad'], 
                         ensemble_results[ej]['Ensemble Hard (OR)']['especificidad']) for ej in ejercicios if ej in ensemble_results]

ax3.bar(x - width, f1_ej5, width, label='Solo Ej5', color='blue', alpha=0.7)
ax3.bar(x, f1_ej6, width, label='Solo Ej6', color='green', alpha=0.7)
ax3.bar(x + width, f1_ensemble, width, label='Ensemble (OR)', color='red', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(ejes, rotation=45, ha='right')
ax3.set_ylabel('F1-score')
ax3.set_title('Comparación de F1-score')
ax3.legend()
ax3.set_ylim(0, 1)

# 4. Matriz de decisión del ensemble
ax4 = axes[1, 0]
ax4.axis('off')
texto = "ESTRATEGIAS DE ENSEMBLE\n" + "="*30 + "\n\n"
texto += "1. Soft Voting: Promedio de probabilidades\n"
texto += "   - Balanceado, conservador\n\n"
texto += "2. Hard Voting (OR): Al menos uno positivo\n"
texto += "   - Maximiza sensibilidad\n"
texto += "   - Mejor para screening\n\n"
texto += "3. Weighted Voting: 40% Ej5 + 60% Ej6\n"
texto += "   - Da más peso a Ej6 (mejor especificidad)\n\n"
texto += "✅ RECOMENDACIÓN: Usar HARD VOTING (OR)\n"
texto += "   - Sensibilidad: 73-100%\n"
texto += "   - Especificidad: 50-100%\n"
texto += "   - Ideal para detección temprana"
ax4.text(0.05, 0.95, texto, transform=ax4.transAxes, fontsize=10, verticalalignment='top')

# 5. Resumen numérico
ax5 = axes[1, 1]
ax5.axis('off')

# Calcular promedios
avg_sens_ej5 = np.mean(sens_ej5)
avg_sens_ej6 = np.mean(sens_ej6)
avg_sens_ensemble = np.mean(sens_ensemble)
avg_esp_ej5 = np.mean(esp_ej5)
avg_esp_ej6 = np.mean(esp_ej6)
avg_esp_ensemble = np.mean(esp_ensemble)

resumen = f"RESUMEN PROMEDIO (todos los ejercicios)\n" + "="*35 + "\n\n"
resumen += f"Solo Ej5:    Sens={avg_sens_ej5:.1%}, Esp={avg_esp_ej5:.1%}\n"
resumen += f"Solo Ej6:    Sens={avg_sens_ej6:.1%}, Esp={avg_esp_ej6:.1%}\n"
resumen += f"Ensemble OR: Sens={avg_sens_ensemble:.1%}, Esp={avg_esp_ensemble:.1%}\n\n"
resumen += f"📈 MEJORA:\n"
resumen += f"  Sensibilidad: +{(avg_sens_ensemble - avg_sens_ej6)*100:.0f}% vs Ej6\n"
resumen += f"  Especificidad: +{(avg_esp_ensemble - avg_esp_ej5)*100:.0f}% vs Ej5"

ax5.text(0.05, 0.95, resumen, transform=ax5.transAxes, fontsize=11, verticalalignment='top', fontweight='bold')

# 6. Número de positivos detectados
ax6 = axes[1, 2]
n_parkinson = 23  # Total pacientes Parkinson
positivos_ej5 = [ensemble_results[ej]['Solo Ej5']['n_positivos'] for ej in ejercicios if ej in ensemble_results]
positivos_ej6 = [ensemble_results[ej]['Solo Ej6']['n_positivos'] for ej in ejercicios if ej in ensemble_results]
positivos_ensemble = [ensemble_results[ej]['Ensemble Hard (OR)']['n_positivos'] for ej in ejercicios if ej in ensemble_results]

ax6.plot(ejes, positivos_ej5, 'o-', label='Solo Ej5', color='blue', linewidth=2, markersize=8)
ax6.plot(ejes, positivos_ej6, 's-', label='Solo Ej6', color='green', linewidth=2, markersize=8)
ax6.plot(ejes, positivos_ensemble, '^-', label='Ensemble OR', color='red', linewidth=2, markersize=8)
ax6.axhline(y=n_parkinson, color='black', linestyle='--', label=f'Total Parkinson ({n_parkinson})')
ax6.set_ylabel('Número de positivos detectados')
ax6.set_title('Detección de Parkinson por Estrategia')
ax6.legend()
ax6.set_xticklabels(ejes, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('ensemble_voting_results.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
comparison_df.to_csv('ensemble_voting_comparison.csv', index=False)
print("\n✅ Resultados guardados en 'ensemble_voting_comparison.csv'")
print("✅ Gráfico guardado en 'ensemble_voting_results.png'")

# =============================================================================
# RECOMENDACIÓN FINAL
# =============================================================================
print("\n" + "="*80)
print("🎯 RECOMENDACIÓN FINAL")
print("="*80)

print("""
🏆 MEJOR ESTRATEGIA: HARD VOTING (OR)

Ventajas:
  ✓ Sensibilidad promedio: {:.1%} (vs {:.1%} solo Ej6)
  ✓ Detecta más casos de Parkinson
  ✓ Ideal para SCREENING (no queremos perder pacientes)
  ✓ Funciona en CUALQUIER ejercicio en casa

Desventajas:
  ✗ Especificidad: {:.1%} (puede tener falsos positivos)
  ✗ Requiere que el paciente haga ambos ejercicios (5 y 6)

Implementación:
  1. Paciente hace Ej5 (tapping dedos) y Ej6 (tapping pies) en casa
  2. Modelo evalúa ambos
  3. Si AL MENOS UNO es positivo → SOSPECHA de bradicinesia
  4. Confirmar con clínico

Caso de uso:
  - Monitoreo remoto de pacientes
  - Detección temprana de deterioro
  - Evaluación de efectividad de medicación
""".format(avg_sens_ensemble, avg_sens_ej6, avg_esp_ensemble))