# =============================================================================
# analysis/data_leakage/detect_leakage.py
# Detectar fugas de datos entre train y test
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from src.data.loader import BIOCLITEDataset

print("="*80)
print("🔍 ANÁLISIS DE DATA LEAKAGE")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Filtrar ejercicio 6 en contexto supervisado
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

print(f"\n📊 Datos: {len(df_clinic):,} filas")

# =============================================================================
# 1. Verificar solapamiento temporal dentro del mismo sujeto
# =============================================================================
print("\n" + "="*60)
print("1. SOLAPAMIENTO TEMPORAL POR SUJETO")
print("="*60)

WINDOW_SIZE = 100
STEP_SIZES = [25, 50, 75, 100, 150, 200]  # Diferentes solapamientos

leakage_report = []

for step in STEP_SIZES:
    overlap_percent = (1 - step/WINDOW_SIZE) * 100
    n_windows = 0
    unique_windows_per_session = defaultdict(set)
    
    for session in df_clinic['Sesion'].unique():
        df_ses = df_clinic[df_clinic['Sesion'] == session]
        n_samples = len(df_ses)
        
        if n_samples < WINDOW_SIZE:
            continue
        
        # Calcular número de ventanas
        n_wins = (n_samples - WINDOW_SIZE) // step + 1
        n_windows += n_wins
        
        # Calcular cuántas muestras únicas se usan
        used_indices = set()
        for i in range(0, n_samples - WINDOW_SIZE, step):
            for j in range(i, i + WINDOW_SIZE):
                used_indices.add(j)
        
        unique_windows_per_session[session] = len(used_indices)
    
    total_unique_samples = sum(unique_windows_per_session.values())
    total_available_samples = len(df_clinic)
    
    leakage_ratio = total_unique_samples / total_available_samples
    
    leakage_report.append({
        'step_size': step,
        'overlap_percent': overlap_percent,
        'n_windows': n_windows,
        'unique_samples_used': total_unique_samples,
        'leakage_ratio': leakage_ratio
    })
    
    print(f"  Step={step} ({overlap_percent:.0f}% overlap): {n_windows} ventanas, {leakage_ratio:.1%} de datos únicos")

leakage_df = pd.DataFrame(leakage_report)
print("\n📊 Resumen:")
print(leakage_df.to_string())

# Figura: leakage vs overlap
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(leakage_df['overlap_percent'], leakage_df['leakage_ratio'], 'o-', linewidth=2, markersize=8)
ax.axhline(y=1.0, color='red', linestyle='--', label='100% (sin muestras únicas)')
ax.set_xlabel('Solapamiento entre ventanas (%)')
ax.set_ylabel('Proporción de muestras únicas usadas')
ax.set_title('Data Leakage: Reutilización de muestras por solapamiento')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/data_leakage/leakage_analysis.png', dpi=300)
plt.close()
print("\n✅ Figura guardada: analysis/data_leakage/leakage_analysis.png")

# =============================================================================
# 2. Verificar que sujetos no se mezclan entre train/test
# =============================================================================
print("\n" + "="*60)
print("2. SEPARACIÓN POR SUJETO (LOSO)")
print("="*60)

subjects = df_clinic['subject_id'].unique()
print(f"Total sujetos: {len(subjects)}")

# Verificar distribución de clases por sujeto
subject_info = []
for subject in subjects:
    df_subj = df_clinic[df_clinic['subject_id'] == subject]
    updrs_values = df_subj['UPDRS'].unique()
    main_updrs = updrs_values[0] if len(updrs_values) > 0 else None
    
    subject_info.append({
        'subject': subject,
        'n_samples': len(df_subj),
        'n_sessions': df_subj['Sesion'].nunique(),
        'updrs': main_updrs,
        'has_bradykinesia': 1 if main_updrs not in [0, 99] else 0
    })

subject_df = pd.DataFrame(subject_info)
print(f"\nSujetos con bradicinesia: {subject_df['has_bradykinesia'].sum()}")
print(f"Sujetos sin bradicinesia: {len(subject_df) - subject_df['has_bradykinesia'].sum()}")

# Verificar si algún sujeto tiene múltiples UPDRS (inconsistencia)
print("\n🔍 Verificando consistencia de UPDRS por sujeto:")
inconsistent = []
for subject in subjects:
    df_subj = df_clinic[df_clinic['subject_id'] == subject]
    updrs_unique = df_subj['UPDRS'].unique()
    if len(updrs_unique) > 2 or (len(updrs_unique) == 2 and 99 not in updrs_unique):
        inconsistent.append({'subject': subject, 'updrs_values': updrs_unique.tolist()})

if inconsistent:
    print("⚠️ SUJETOS CON UPDRS INCONSISTENTE:")
    for inc in inconsistent:
        print(f"  {inc['subject']}: {inc['updrs_values']}")
else:
    print("✅ Todos los sujetos tienen UPDRS consistente")

# =============================================================================
# 3. Análisis de correlación temporal
# =============================================================================
print("\n" + "="*60)
print("3. CORRELACIÓN TEMPORAL ENTRE VENTANAS")
print("="*60)

# Tomar un sujeto de ejemplo
example_subject = subject_df[subject_df['has_bradykinesia'] == 1]['subject'].iloc[0]
df_example = df_clinic[df_clinic['subject_id'] == example_subject].copy()

# Calcular señal de aceleración
acc_mag = np.sqrt(df_example['Acc_X']**2 + df_example['Acc_Y']**2 + df_example['Acc_Z']**2)

# Calcular autocorrelación
from scipy.signal import correlate
autocorr = correlate(acc_mag, acc_mag, mode='full')
autocorr = autocorr[len(autocorr)//2:]

# Figura de autocorrelación
fig, ax = plt.subplots(figsize=(12, 5))
lags = np.arange(len(autocorr)) / 50  # segundos
ax.plot(lags[:500], autocorr[:500])
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Lag (segundos)')
ax.set_ylabel('Autocorrelación')
ax.set_title(f'Autocorrelación de la señal - Sujeto {example_subject}')
ax.grid(True, alpha=0.3)

# Marcar el tiempo de ventana
ax.axvline(x=WINDOW_SIZE/50, color='red', linestyle='--', label=f'Ventana ({WINDOW_SIZE/50}s)')
ax.legend()
plt.tight_layout()
plt.savefig('analysis/data_leakage/temporal_correlation.png', dpi=300)
plt.close()
print("✅ Figura guardada: analysis/data_leakage/temporal_correlation.png")

print("\n" + "="*60)
print("📊 CONCLUSIONES DATA LEAKAGE")
print("="*60)
print("""
Para evitar data leakage:
1. Usar step_size = window_size (0% overlap) o step_size > window_size/2
2. Siempre separar por sujeto (LOSO)
3. Verificar que un sujeto no aparezca en train y test
4. No usar ventanas solapadas si se necesita independencia
""")