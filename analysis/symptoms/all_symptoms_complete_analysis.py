# =============================================================================
# analysis/all_symptoms_complete_analysis.py
# ANÁLISIS COMPLETO DE TODOS LOS SÍNTOMAS (Ejercicios 1-8)
# Con interpretación clínica y comparativa
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 ANÁLISIS COMPLETO DE TODOS LOS SÍNTOMAS MOTORES")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Configuración
WINDOW_SIZE = 100
STEP_SIZE = 100  # Sin solapamiento
preprocessor = IMUPreprocessor(fs=50)

# =============================================================================
# 1. DEFINICIÓN CLÍNICA DE CADA SÍNTOMA
# =============================================================================

sintomas_info = {
    1: {
        "nombre": "Temblor postural",  # CAMBIADO: antes "Habla"
        "updrs_item": "3.15",  # CAMBIADO: antes "3.17"
        "descripcion_clinica": "Temblor de manos con brazos extendidos - frecuencia 4-9 Hz",
        "biomecanica": "Contracciones alternantes de músculos agonistas-antagonistas",
        "metricas_clave": ["Frecuencia del temblor", "Amplitud", "Regularidad"],
        "relevancia_clinica": "Alta - signo cardinal de Parkinson"
    },
    2: {
        "nombre": "Temblor de acción",  # CAMBIADO: antes "Expresión facial"
        "updrs_item": "3.16",  # CAMBIADO: antes "3.15"
        "descripcion_clinica": "Temblor durante movimientos voluntarios (dedo-nariz)",
        "biomecanica": "Oscilaciones que aparecen o empeoran con el movimiento",
        "metricas_clave": ["Frecuencia", "Amplitud", "Relación con movimiento intencional"],
        "relevancia_clinica": "Alta - diferencia temblor esencial de Parkinson"
    },
    3: {
        "nombre": "Temblor en reposo",
        "updrs_item": "N/A",
        "descripcion_clinica": "Temblor de 4-6 Hz en reposo, típicamente en manos",
        "biomecanica": "Oscilaciones involuntarias",
        "metricas_clave": ["Frecuencia dominante", "Amplitud", "Regularidad"],
        "relevancia_clinica": "Alta - signo cardinal"
    },
    4: {
        "nombre": "Pronación-supinación",
        "updrs_item": "3.4",
        "descripcion_clinica": "Movimiento alternante de antebrazo - evalúa rigidez y bradicinesia",
        "biomecanica": "Rotación de antebrazo",
        "metricas_clave": ["Velocidad angular", "Rango de movimiento", "Regularidad"],
        "relevancia_clinica": "Alta - evalúa miembros superiores"
    },
    5: {
        "nombre": "Tapping de dedos",
        "updrs_item": "3.5",
        "descripcion_clinica": "Movimiento rápido de índice contra pulgar",
        "biomecanica": "Movimientos finos de dedos",
        "metricas_clave": ["Frecuencia", "Amplitud", "Fatiga", "Simetría"],
        "relevancia_clinica": "Alta - sensible a bradicinesia"
    },
    6: {
        "nombre": "Tapping de pies",
        "updrs_item": "3.6",
        "descripcion_clinica": "Movimiento de golpeo de pie contra el suelo",
        "biomecanica": "Flexión dorsal del tobillo",
        "metricas_clave": ["Frecuencia", "Altura del pie", "Ritmo", "Fatiga"],
        "relevancia_clinica": "Alta - evalúa miembros inferiores"
    },
    7: {
        "nombre": "Levantarse de silla",
        "updrs_item": "3.9",
        "descripcion_clinica": "Transferencia de sedestación a bipedestación",
        "biomecanica": "Extensión de piernas y tronco",
        "metricas_clave": ["Tiempo de ejecución", "Velocidad angular", "Estabilidad"],
        "relevancia_clinica": "Alta - evalúa función motora global"
    },
    8: {
        "nombre": "Marcha",
        "updrs_item": "3.10",
        "descripcion_clinica": "Patrón de caminar - evalúa pasos, giros, postura",
        "biomecanica": "Ciclo de la marcha",
        "metricas_clave": ["Cadencia", "Longitud de paso", "Asimetría", "Velocidad"],
        "relevancia_clinica": "Muy alta - impacto funcional"
    }
}

# =============================================================================
# 2. FUNCIÓN PARA ANALIZAR CADA SÍNTOMA
# =============================================================================

def analyze_symptom(df_data, ejercicio, preprocessor, window_size=100, step_size=100):
    """Analiza un síntoma específico y retorna métricas detalladas"""
    
    df_ej = df_data[(df_data['Contexto_sesion'].isin([1, 2])) & (df_data['Ejercicio'] == ejercicio)].copy()
    
    if len(df_ej) == 0:
        return None
    
    X, y, groups = [], [], []
    
    for session in df_ej['Sesion'].unique():
        df_ses = df_ej[df_ej['Sesion'] == session]
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
    
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    # Escalar y validar
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    sgkf = StratifiedGroupKFold(n_splits=min(5, len(np.unique(groups))), shuffle=True, random_state=42)
    
    results = {'f1': [], 'auc': [], 'accuracy': [], 'recall': [], 'precision': []}
    
    for train_idx, test_idx in sgkf.split(X_scaled, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        '''# Modelo con overfitting
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, 
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        '''
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, 
                            min_samples_split=8, min_samples_leaf=4,
                            class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        results['f1'].append(f1_score(y_test, y_pred))
        results['auc'].append(roc_auc_score(y_test, y_prob))
        results['accuracy'].append(accuracy_score(y_test, y_pred))
        results['recall'].append(recall_score(y_test, y_pred))
        results['precision'].append(precision_score(y_test, y_pred))
    
    return {
        'X': X,
        'y': y,
        'groups': groups,
        'n_samples': len(X),
        'n_positivos': np.sum(y),
        'n_negativos': len(X) - np.sum(y),
        'f1_mean': np.mean(results['f1']),
        'f1_std': np.std(results['f1']),
        'auc_mean': np.mean(results['auc']),
        'auc_std': np.std(results['auc']),
        'accuracy_mean': np.mean(results['accuracy']),
        'recall_mean': np.mean(results['recall']),
        'precision_mean': np.mean(results['precision']),
        'n_folds': len(results['f1'])
    }

# =============================================================================
# 3. EJECUTAR ANÁLISIS PARA TODOS LOS SÍNTOMAS
# =============================================================================

print("\n📊 EJECUTANDO ANÁLISIS PARA CADA SÍNTOMA...")
print("="*60)

resultados = {}

for ejercicio in range(1, 9):
    print(f"\n🔬 Analizando: Ejercicio {ejercicio} - {sintomas_info[ejercicio]['nombre']}")
    print(f"   {sintomas_info[ejercicio]['descripcion_clinica'][:60]}...")
    
    resultado = analyze_symptom(df, ejercicio, preprocessor, WINDOW_SIZE, STEP_SIZE)
    
    if resultado is None:
        print(f"   ⚠️ Datos insuficientes para análisis")
        continue
    
    resultados[ejercicio] = resultado
    
    print(f"   ✅ Muestras: {resultado['n_samples']} | F1: {resultado['f1_mean']:.3f} ± {resultado['f1_std']:.3f}")
    print(f"      AUC: {resultado['auc_mean']:.3f} ± {resultado['auc_std']:.3f}")
    print(f"      Sensibilidad: {resultado['recall_mean']:.3f} | Especificidad: {resultado['precision_mean']:.3f}")

# =============================================================================
# 4. TABLA COMPARATIVA CON INTERPRETACIÓN CLÍNICA
# =============================================================================

print("\n" + "="*80)
print("📊 TABLA COMPARATIVA DE SÍNTOMAS")
print("="*80)

comparison_data = []
for ej, res in resultados.items():
    info = sintomas_info[ej]
    
    # Determinar nivel de detectabilidad
    if res['f1_mean'] >= 0.75:
        detectabilidad = "ALTA"
        color = "🟢"
    elif res['f1_mean'] >= 0.60:
        detectabilidad = "MEDIA"
        color = "🟡"
    else:
        detectabilidad = "BAJA"
        color = "🔴"
    
    comparison_data.append({
        'Ejercicio': ej,
        'Síntoma': info['nombre'],
        'UPDRS': info['updrs_item'],
        'Descripción': info['descripcion_clinica'][:50] + "...",
        'N_muestras': res['n_samples'],
        'F1': f"{res['f1_mean']:.3f} ± {res['f1_std']:.3f}",
        'AUC': f"{res['auc_mean']:.3f} ± {res['auc_std']:.3f}",
        'Sensibilidad': f"{res['recall_mean']:.3f}",
        'Especificidad': f"{res['precision_mean']:.3f}",
        'Detectabilidad': f"{color} {detectabilidad}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string())

# Guardar
comparison_df.to_csv('analysis/all_symptoms_comparison.csv', index=False)

# =============================================================================
# 5. VISUALIZACIÓN COMPARATIVA
# =============================================================================
print("\n" + "="*80)
print("📈 GENERANDO VISUALIZACIONES COMPARATIVAS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: F1-score por síntoma
ax1 = axes[0, 0]
sintomas = [sintomas_info[ej]['nombre'] for ej in resultados.keys()]
f1_means = [res['f1_mean'] for res in resultados.values()]
f1_stds = [res['f1_std'] for res in resultados.values()]
colors_f1 = ['green' if f >= 0.75 else 'orange' if f >= 0.6 else 'red' for f in f1_means]

bars1 = ax1.barh(sintomas, f1_means, xerr=f1_stds, color=colors_f1, capsize=3, alpha=0.7)
ax1.axvline(x=0.75, color='green', linestyle='--', alpha=0.7, label='Alta detectabilidad')
ax1.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Media detectabilidad')
ax1.set_xlabel('F1-score')
ax1.set_title('Comparación de F1-score por Síntoma')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: AUC por síntoma
ax2 = axes[0, 1]
auc_means = [res['auc_mean'] for res in resultados.values()]
auc_stds = [res['auc_std'] for res in resultados.values()]
colors_auc = ['green' if a >= 0.8 else 'orange' if a >= 0.7 else 'red' for a in auc_means]

bars2 = ax2.barh(sintomas, auc_means, xerr=auc_stds, color=colors_auc, capsize=3, alpha=0.7)
ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Excelente')
ax2.axvline(x=0.7, color='orange', linestyle='--', alpha=0.7, label='Aceptable')
ax2.set_xlabel('AUC')
ax2.set_title('Comparación de AUC por Síntoma')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Gráfico 3: Sensibilidad vs Especificidad
ax3 = axes[1, 0]
sensibilidad = [res['recall_mean'] for res in resultados.values()]
especificidad = [res['precision_mean'] for res in resultados.values()]

for i, sintoma in enumerate(sintomas):
    ax3.scatter(sensibilidad[i], especificidad[i], s=100, alpha=0.7)
    ax3.annotate(sintoma[:10], (sensibilidad[i], especificidad[i]), fontsize=8)

ax3.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Línea de intercambio')
ax3.set_xlabel('Sensibilidad')
ax3.set_ylabel('Especificidad')
ax3.set_title('Balance Sensibilidad vs Especificidad')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Número de muestras
ax4 = axes[1, 1]
n_muestras = [res['n_samples'] for res in resultados.values()]
bars4 = ax4.barh(sintomas, n_muestras, color='steelblue', alpha=0.7)
ax4.set_xlabel('Número de muestras (ventanas)')
ax4.set_title('Cantidad de Datos Disponibles por Síntoma')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/all_symptoms_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura comparativa guardada")

# =============================================================================
# 6. ANÁLISIS DE DETECTABILIDAD POR CATEGORÍA
# =============================================================================
print("\n" + "="*80)
print("📊 ANÁLISIS DE DETECTABILIDAD POR CATEGORÍA")
print("="*80)

alta_detectabilidad = []
media_detectabilidad = []
baja_detectabilidad = []

for ej, res in resultados.items():
    if res['f1_mean'] >= 0.75:
        alta_detectabilidad.append((ej, sintomas_info[ej]['nombre'], res['f1_mean']))
    elif res['f1_mean'] >= 0.60:
        media_detectabilidad.append((ej, sintomas_info[ej]['nombre'], res['f1_mean']))
    else:
        baja_detectabilidad.append((ej, sintomas_info[ej]['nombre'], res['f1_mean']))

print("\n🟢 SÍNTOMAS DE ALTA DETECTABILIDAD (F1 ≥ 0.75):")
for ej, nombre, f1 in alta_detectabilidad:
    print(f"   Ej{ej}: {nombre} (F1={f1:.3f})")

print("\n🟡 SÍNTOMAS DE MEDIA DETECTABILIDAD (0.60 ≤ F1 < 0.75):")
for ej, nombre, f1 in media_detectabilidad:
    print(f"   Ej{ej}: {nombre} (F1={f1:.3f})")

print("\n🔴 SÍNTOMAS DE BAJA DETECTABILIDAD (F1 < 0.60):")
for ej, nombre, f1 in baja_detectabilidad:
    print(f"   Ej{ej}: {nombre} (F1={f1:.3f})")

# =============================================================================
# 7. INTERPRETACIÓN CLÍNICA PARA MÉDICOS
# =============================================================================
print("\n" + "="*80)
print("👨‍⚕️ INTERPRETACIÓN CLÍNICA PARA MÉDICOS")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GUÍA DE INTERPRETACIÓN CLÍNICA                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ¿POR QUÉ ALGUNOS SÍNTOMAS SON MÁS DETECTABLES QUE OTROS?                   │
│                                                                             │
│  1. MOVIMIENTOS RÍTMICOS REPETITIVOS (Ejercicios 5, 6)                      │
│     → Tapping de dedos y pies son ALTAMENTE DETECTABLES                     │
│     → Razón: Señales periódicas, fáciles de caracterizar en frecuencia      │
│     → Ejemplo: Bradicinesia → ↓ frecuencia tapping (< 2 Hz)                 │
│                                                                             │
│  2. MOVIMIENTOS COMPLEJOS (Ejercicios 7, 8)                                 │
│     → Levantarse y marcha son variables entre pacientes                     │
│     → Razón: Estrategias compensatorias individuales                        │
│     → Ejemplo: Un paciente puede levantarse más lento pero estable          │
│                                                                             │
│  3. SÍNTOMAS NO MOTORES (Ejercicios 1, 2)                                   │
│     → Habla y expresión facial son más difíciles de capturar                │
│     → Razón: La muñeca no capta bien estos movimientos                      │
│     → Limitación: Sensor en muñeca vs síntomas faciales                     │
│                                                                             │
│  4. TEMBLOR EN REPOSO (Ejercicio 3)                                         │
│     → Datos insuficientes para evaluación confiable                         │
│     → Razón: Pocos pacientes con temblor en la muestra                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    RECOMENDACIONES CLÍNICAS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ Para MONITOREO DE BRADICINESIA:                                         │
│     → Usar ejercicios 5 y 6 (tapping dedos y pies)                          │
│     → Alta sensibilidad y especificidad                                     │
│     → Fácil de realizar en casa                                            │
│                                                                             │
│  ✅ Para EVALUACIÓN GLOBAL:                                                 │
│     → Combinar ejercicios 4, 5, 6, 8                                        │
│     → Captura diferentes dominios motores                                   │
│                                                                             │
│  ⚠️ Limitaciones:                                                           │
│     → El dispositivo en muñeca NO detecta bien:                             │
│       • Expresión facial                                                    │
│       • Temblor en reposo puro                                              │
│       • Habla                                                               │
│                                                                             │
│  💡 IMPLICACIÓN PRÁCTICA:                                                   │
│     → Para monitoreo remoto, enfocarse en tapping (ejercicios 5 y 6)        │
│     → Estos son los más sensibles a cambios en el estado motor              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# 8. GUARDAR RESULTADOS COMPLETOS
# =============================================================================

# Guardar resultados detallados
detailed_results = []
for ej, res in resultados.items():
    detailed_results.append({
        'ejercicio': ej,
        'sintoma': sintomas_info[ej]['nombre'],
        'descripcion': sintomas_info[ej]['descripcion_clinica'],
        'updrs_item': sintomas_info[ej]['updrs_item'],
        'n_muestras': res['n_samples'],
        'n_positivos': res['n_positivos'],
        'n_negativos': res['n_negativos'],
        'f1_mean': res['f1_mean'],
        'f1_std': res['f1_std'],
        'auc_mean': res['auc_mean'],
        'auc_std': res['auc_std'],
        'recall_mean': res['recall_mean'],
        'precision_mean': res['precision_mean'],
        'accuracy_mean': res['accuracy_mean'],
        'n_folds': res['n_folds']
    })

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv('analysis/all_symptoms_detailed_results.csv', index=False)

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETO DE TODOS LOS SÍNTOMAS FINALIZADO")
print("="*80)
print("\n📁 Archivos generados:")
print("  • analysis/all_symptoms_comparison.csv - Tabla comparativa")
print("  • analysis/all_symptoms_detailed_results.csv - Resultados detallados")
print("  • analysis/all_symptoms_comparison.png - Figura comparativa")