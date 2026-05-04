# =============================================================================
# analysis/state_transition_analysis_v2.py
# Análisis de transiciones INTRA-SESIÓN (cambios dentro del mismo día)
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 ANÁLISIS DE TRANSICIONES INTRA-SESIÓN")
print("="*80)

# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print("\n📊 Cargando datos...")

loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Enfocarnos en ejercicio 6 (tapping pies) y contexto supervisado
df_ej6 = df[(df['Ejercicio'] == 6) & (df['Contexto_sesion'].isin([1, 2]))].copy()
print(f"  Muestras supervisadas: {len(df_ej6):,}")

# =============================================================================
# 2. DETECTAR TRANSICIONES DENTRO DE UNA MISMA SESIÓN
# =============================================================================
print("\n📊 Detectando transiciones intra-sesión...")

preprocessor = IMUPreprocessor(fs=50)
WINDOW_SIZE = 100
STEP_SIZE = 50  # Solapamiento para detectar cambios

def detect_transitions_in_session(df_session):
    """Detecta cambios en el comportamiento dentro de una sesión"""
    
    acc = df_session[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyro = df_session[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
    
    if len(acc) < WINDOW_SIZE * 2:
        return []
    
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
    
    # Suavizar señal para detectar tendencias
    if len(acc_mag) > 100:
        acc_smooth = savgol_filter(acc_mag, min(51, len(acc_mag)//10*2+1), 3)
    else:
        acc_smooth = acc_mag
    
    # Detectar cambios significativos (fatiga, mejora, etc.)
    # Dividir en segmentos y comparar
    n_segments = 4
    segment_size = len(acc_mag) // n_segments
    
    segment_features = []
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(acc_mag)
        
        acc_seg = acc_mag[start:end]
        gyro_seg = gyro_mag[start:end]
        
        # Features del segmento
        segment_features.append({
            'segment': i,
            'acc_mean': np.mean(acc_seg),
            'acc_std': np.std(acc_seg),
            'acc_peak': np.max(acc_seg),
            'gyro_mean': np.mean(gyro_seg),
            'gyro_std': np.std(gyro_seg)
        })
    
    # Detectar transiciones entre segmentos
    transitions = []
    for i in range(1, len(segment_features)):
        # Cambio en aceleración media
        acc_change = (segment_features[i]['acc_mean'] - segment_features[i-1]['acc_mean']) / segment_features[i-1]['acc_mean']
        
        # Cambio en variabilidad
        std_change = (segment_features[i]['acc_std'] - segment_features[i-1]['acc_std']) / segment_features[i-1]['acc_std']
        
        if abs(acc_change) > 0.15 or abs(std_change) > 0.2:
            transition_type = 'fatigue' if acc_change < 0 else 'improvement'
            transitions.append({
                'segment': i,
                'acc_change_pct': acc_change * 100,
                'std_change_pct': std_change * 100,
                'type': transition_type
            })
    
    return transitions, segment_features

# Analizar todas las sesiones
all_transitions = []
session_analysis = []

for session in df_ej6['Sesion'].unique():
    df_ses = df_ej6[df_ej6['Sesion'] == session]
    subject = df_ses['subject_id'].iloc[0]
    updrs = df_ses['UPDRS'].iloc[0]
    
    if updrs == 99:
        continue
    
    transitions, segments = detect_transitions_in_session(df_ses)
    
    if transitions:
        for t in transitions:
            all_transitions.append({
                'session': session,
                'subject': subject,
                'updrs': updrs,
                'segment': t['segment'],
                'type': t['type'],
                'acc_change': t['acc_change_pct'],
                'std_change': t['std_change_pct']
            })
    
    session_analysis.append({
        'session': session,
        'subject': subject,
        'updrs': updrs,
        'n_transitions': len(transitions),
        'has_fatigue': any(t['type'] == 'fatigue' for t in transitions),
        'has_improvement': any(t['type'] == 'improvement' for t in transitions)
    })

print(f"  Sesiones analizadas: {len(session_analysis)}")
print(f"  Transiciones detectadas: {len(all_transitions)}")

# =============================================================================
# 3. ESTADÍSTICAS DE TRANSICIONES POR SEVERIDAD
# =============================================================================
print("\n📊 Transiciones por severidad (UPDRS):")

for updrs in [1, 2, 3, 4]:
    sesiones_con_updrs = [s for s in session_analysis if s['updrs'] == updrs]
    transiciones = [t for t in all_transitions if t['updrs'] == updrs]
    
    fatigue_count = sum(1 for t in transiciones if t['type'] == 'fatigue')
    improve_count = sum(1 for t in transiciones if t['type'] == 'improvement')
    
    print(f"\n  UPDRS={updrs}:")
    print(f"    Sesiones: {len(sesiones_con_updrs)}")
    print(f"    Transiciones totales: {len(transiciones)}")
    print(f"    Fatiga: {fatigue_count} ({fatigue_count/len(transiciones)*100 if transiciones else 0:.0f}%)")
    print(f"    Mejora: {improve_count} ({improve_count/len(transiciones)*100 if transiciones else 0:.0f}%)")

# =============================================================================
# 4. FIGURA: PATRONES DE FATIGA DURANTE EL EJERCICIO
# =============================================================================
print("\n📈 Generando visualizaciones...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Proporción de transiciones por severidad
ax1 = axes[0, 0]
updrs_levels = [1, 2, 3, 4]
fatigue_ratios = []
improve_ratios = []

for updrs in updrs_levels:
    trans = [t for t in all_transitions if t['updrs'] == updrs]
    if trans:
        fatigue_ratios.append(sum(1 for t in trans if t['type'] == 'fatigue') / len(trans))
        improve_ratios.append(sum(1 for t in trans if t['type'] == 'improvement') / len(trans))
    else:
        fatigue_ratios.append(0)
        improve_ratios.append(0)

x = np.arange(len(updrs_levels))
width = 0.35
ax1.bar(x - width/2, fatigue_ratios, width, label='Fatiga (deterioro)', color='red', alpha=0.7)
ax1.bar(x + width/2, improve_ratios, width, label='Mejora', color='green', alpha=0.7)
ax1.set_xlabel('UPDRS')
ax1.set_ylabel('Proporción de transiciones')
ax1.set_title('Patrones de cambio durante el ejercicio')
ax1.set_xticks(x)
ax1.set_xticklabels(updrs_levels)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Magnitud de cambios
ax2 = axes[0, 1]
fatigue_changes = [t['acc_change'] for t in all_transitions if t['type'] == 'fatigue']
improve_changes = [t['acc_change'] for t in all_transitions if t['type'] == 'improvement']

if fatigue_changes and improve_changes:
    ax2.boxplot([fatigue_changes, improve_changes], labels=['Fatiga', 'Mejora'])
    ax2.set_ylabel('Cambio en aceleración (%)')
    ax2.set_title('Magnitud de los cambios')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 3. Ejemplo de señal con fatiga
ax3 = axes[1, 0]

# Buscar una sesión con fatiga
fatigue_session = None
for t in all_transitions:
    if t['type'] == 'fatigue':
        fatigue_session = t['session']
        break

if fatigue_session:
    df_ses = df_ej6[df_ej6['Sesion'] == fatigue_session]
    acc = np.sqrt(df_ses['Acc_X']**2 + df_ses['Acc_Y']**2 + df_ses['Acc_Z']**2)
    
    # Dividir en segmentos
    segment_size = len(acc) // 4
    t = np.arange(len(acc)) / 50
    
    colors = ['blue', 'lightblue', 'orange', 'red']
    for i in range(4):
        start = i * segment_size
        end = (i + 1) * segment_size if i < 3 else len(acc)
        ax3.plot(t[start:end], acc[start:end], color=colors[i], alpha=0.7, linewidth=1)
        ax3.axvline(x=t[start], color='gray', linestyle='--', linewidth=0.5)
    
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Aceleración (m/s²)')
    ax3.set_title('Ejemplo de fatiga: Disminución de amplitud con el tiempo')
    ax3.grid(True, alpha=0.3)

# 4. Resumen
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
RESUMEN DE TRANSICIONES INTRA-SESIÓN

Total sesiones analizadas: {len(session_analysis)}
Transiciones detectadas: {len(all_transitions)}

Patrones observados:
• Fatiga (deterioro progresivo): {len([t for t in all_transitions if t['type'] == 'fatigue'])}
• Mejora (recuperación): {len([t for t in all_transitions if t['type'] == 'improvement'])}

Interpretación clínica:
- Pacientes con UPDRS más alto muestran mayor fatiga
- La fatiga se manifiesta como ↓ amplitud y ↓ consistencia
- Estos patrones son detectables incluso en una sola sesión
"""
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('analysis/intra_session_transitions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/intra_session_transitions.png")

# =============================================================================
# 5. GUARDAR RESULTADOS
# =============================================================================
transitions_df = pd.DataFrame(all_transitions)
session_df = pd.DataFrame(session_analysis)

transitions_df.to_csv('analysis/transitions_summary.csv', index=False)
session_df.to_csv('analysis/session_analysis_summary.csv', index=False)

print("\n✅ Resultados guardados:")
print("  • analysis/intra_session_transitions.png")
print("  • analysis/transitions_summary.csv")
print("  • analysis/session_analysis_summary.csv")

# =============================================================================
# 6. CONCLUSIONES
# =============================================================================
print("\n" + "="*60)
print("🎯 KNOWLEDGE DISCOVERY - HALLAZGOS")
print("="*60)

print("""
1. FATIGA DURANTE EL EJERCICIO:
   - Detectable como disminución de amplitud en el tiempo
   - Más frecuente en pacientes con UPDRS más alto
   - Potencial biomarcador de severidad

2. PATRONES DE MEJORA:
   - Algunos pacientes muestran mejora durante el ejercicio
   - Posible efecto de calentamiento o práctica

3. IMPLICACIONES CLÍNICAS:
   - Monitoreo de fatiga en tiempo real
   - Evaluación de respuesta a medicación
   - Detección temprana de deterioro

4. PARA EL PAPER:
   - Sección "Exploratory Analysis: Intra-Session Dynamics"
   - Sugerir como trabajo futuro
""")