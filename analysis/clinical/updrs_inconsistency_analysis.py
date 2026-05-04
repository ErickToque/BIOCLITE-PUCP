# =============================================================================
# analysis/updrs_inconsistency_analysis.py
# Análisis de UPDRS inconsistente por sujeto
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
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 ANÁLISIS DE UPDRS INCONSISTENTE POR SUJETO")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

print(f"\n📊 Datos: {len(df_clinic):,} filas")

# =============================================================================
# 1. ANALIZAR DISTRIBUCIÓN DE UPDRS POR SUJETO
# =============================================================================
print("\n" + "="*60)
print("1. DISTRIBUCIÓN DE UPDRS POR SUJETO")
print("="*60)

subject_updrs = {}
for subject in df_clinic['subject_id'].unique():
    df_subj = df_clinic[df_clinic['subject_id'] == subject]
    updrs_values = df_subj['UPDRS'].unique()
    updrs_counts = df_subj['UPDRS'].value_counts().to_dict()
    
    # Filtrar 99 (no disponible)
    updrs_values = [u for u in updrs_values if u != 99]
    
    subject_updrs[subject] = {
        'values': updrs_values,
        'counts': updrs_counts,
        'n_sessions': df_subj['Sesion'].nunique(),
        'n_samples': len(df_subj),
        'main_updrs': max(set(updrs_values), key=updrs_counts.get) if updrs_values else None
    }

# Clasificar sujetos por consistencia
consistent_subjects = []
inconsistent_subjects = []

for subject, info in subject_updrs.items():
    if len(info['values']) == 1:
        consistent_subjects.append(subject)
    else:
        inconsistent_subjects.append(subject)

print(f"\nSujetos consistentes (un solo UPDRS): {len(consistent_subjects)}")
print(f"Sujetos inconsistentes (múltiples UPDRS): {len(inconsistent_subjects)}")

if inconsistent_subjects:
    print("\n📋 SUJETOS INCONSISTENTES:")
    for subject in inconsistent_subjects[:15]:
        info = subject_updrs[subject]
        print(f"  {subject}: UPDRS={info['values']} (sesiones={info['n_sessions']}, muestras={info['n_samples']})")
        
        # Ver patrón temporal
        df_subj = df_clinic[df_clinic['subject_id'] == subject]
        session_updrs = df_subj.groupby('Sesion')['UPDRS'].first()
        print(f"    Sesiones: {dict(session_updrs)}")

# =============================================================================
# 2. VISUALIZAR INCONSISTENCIA
# =============================================================================
print("\n" + "="*60)
print("2. VISUALIZACIÓN DE INCONSISTENCIA")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Distribución de UPDRS por sujeto
ax1 = axes[0, 0]
subject_list = []
updrs_list = []
for subject, info in subject_updrs.items():
    for updrs in info['values']:
        subject_list.append(subject)
        updrs_list.append(updrs)

scatter_data = pd.DataFrame({'subject': subject_list, 'updrs': updrs_list})
sns.stripplot(data=scatter_data, x='subject', y='updrs', ax=ax1, alpha=0.7, jitter=True)
ax1.set_xlabel('Sujeto')
ax1.set_ylabel('UPDRS')
ax1.set_title('Distribución de UPDRS por Sujeto')
ax1.tick_params(axis='x', rotation=90)

# Gráfico 2: Número de sujetos por patrón de inconsistencia
ax2 = axes[0, 1]
pattern_counts = {}
for subject, info in subject_updrs.items():
    pattern = tuple(sorted(info['values']))
    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

patterns = [str(p) for p in pattern_counts.keys()]
counts = list(pattern_counts.values())
ax2.bar(patterns[:10], counts[:10])
ax2.set_xlabel('Patrón de UPDRS')
ax2.set_ylabel('Número de sujetos')
ax2.set_title('Patrones de UPDRS Inconsistentes')
ax2.tick_params(axis='x', rotation=45)

# Gráfico 3: Evolución temporal de un sujeto inconsistente
if inconsistent_subjects:
    ax3 = axes[1, 0]
    example_subject = inconsistent_subjects[0]
    df_example = df_clinic[df_clinic['subject_id'] == example_subject].copy()
    df_example['Dia'] = df_example['Dia_sesion']
    df_example['Contexto'] = df_example['Contexto_sesion'].map({1: 'Clínica Ini', 2: 'Clínica Fin'})
    
    # Agrupar por día y contexto
    daily_updrs = df_example.groupby(['Dia', 'Contexto'])['UPDRS'].first().reset_index()
    
    for _, row in daily_updrs.iterrows():
        color = 'blue' if row['Contexto'] == 'Clínica Ini' else 'red'
        ax3.scatter(row['Dia'], row['UPDRS'], color=color, s=100, alpha=0.7)
        ax3.text(row['Dia'], row['UPDRS'] + 0.1, row['Contexto'][:3], ha='center', fontsize=8)
    
    ax3.set_xlabel('Día del estudio')
    ax3.set_ylabel('UPDRS')
    ax3.set_title(f'Evolución de UPDRS - Sujeto {example_subject}')
    ax3.set_ylim(-0.5, 4.5)
    ax3.set_yticks([0, 1, 2, 3, 4])

# Gráfico 4: Comparación de señales entre días con diferente UPDRS
ax4 = axes[1, 1]
if inconsistent_subjects:
    example_subject = inconsistent_subjects[0]
    df_example = df_clinic[df_clinic['subject_id'] == example_subject].copy()
    
    # Encontrar días con UPDRS bajo y alto
    low_updrs_day = df_example[df_example['UPDRS'] == min(df_example['UPDRS'].unique())]['Sesion'].iloc[0]
    high_updrs_day = df_example[df_example['UPDRS'] == max(df_example['UPDRS'].unique())]['Sesion'].iloc[0]
    
    # Extraer señales
    def get_signal(session):
        df_ses = df_example[df_example['Sesion'] == session]
        acc = np.sqrt(df_ses['Acc_X']**2 + df_ses['Acc_Y']**2 + df_ses['Acc_Z']**2)
        return acc.values[:500]
    
    signal_low = get_signal(low_updrs_day)
    signal_high = get_signal(high_updrs_day)
    
    t = np.arange(len(signal_low)) / 50
    ax4.plot(t, signal_low, 'b-', alpha=0.7, label=f'UPDRS={min(df_example["UPDRS"].unique())}')
    ax4.plot(t, signal_high, 'r-', alpha=0.7, label=f'UPDRS={max(df_example["UPDRS"].unique())}')
    ax4.set_xlabel('Tiempo (s)')
    ax4.set_ylabel('Aceleración (m/s²)')
    ax4.set_title(f'Señales del mismo sujeto - Diferente severidad')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/updrs_inconsistency_analysis.png', dpi=300)
plt.close()
print("✅ Figura guardada: analysis/updrs_inconsistency_analysis.png")

# =============================================================================
# 3. COMPARAR ESTRATEGIAS DE MANEJO
# =============================================================================
print("\n" + "="*60)
print("3. COMPARACIÓN DE ESTRATEGIAS")
print("="*60)

# Extraer features
WINDOW_SIZE = 100
STEP_SIZE = 100
preprocessor = IMUPreprocessor(fs=50)

X, y, groups, updrs_original = [], [], [], []

for session in df_clinic['Sesion'].unique():
    df_ses = df_clinic[df_clinic['Sesion'] == session]
    acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
    
    if len(acc) < WINDOW_SIZE:
        continue
    
    for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
        acc_window = acc[i:i+WINDOW_SIZE]
        gyro_window = gyro[i:i+WINDOW_SIZE]
        
        features = preprocessor.extract_features(acc_window, gyro_window)
        X.append(list(features.values()))
        
        updrs_val = df_ses['UPDRS'].iloc[0]
        updrs_original.append(updrs_val)
        
        groups.append(df_ses['subject_id'].iloc[0])

X = np.array(X)
groups = np.array(groups)
updrs_original = np.array(updrs_original)

print(f"\n📊 Dataset: {X.shape[0]} ventanas")

# Estrategia 1: Usar UPDRS original (cada ventana con su UPDRS)
y_original = (updrs_original > 0) & (updrs_original != 99)
y_original = y_original.astype(int)

# Estrategia 2: Usar UPDRS modal por sujeto
subject_modes = {}
for subject in np.unique(groups):
    subject_mask = groups == subject
    subject_updrs = updrs_original[subject_mask]
    subject_updrs = subject_updrs[subject_updrs != 99]
    if len(subject_updrs) > 0:
        mode = np.bincount(subject_updrs.astype(int)).argmax()
        subject_modes[subject] = mode
    else:
        subject_modes[subject] = 0

y_modal = np.array([1 if subject_modes[g] > 0 else 0 for g in groups])

# Estrategia 3: Usar UPDRS máximo (peor severidad)
subject_max = {}
for subject in np.unique(groups):
    subject_mask = groups == subject
    subject_updrs = updrs_original[subject_mask]
    subject_updrs = subject_updrs[subject_updrs != 99]
    subject_max[subject] = np.max(subject_updrs) if len(subject_updrs) > 0 else 0

y_max = np.array([1 if subject_max[g] > 0 else 0 for g in groups])

# Estrategia 4: Usar UPDRS de la primera sesión (línea base)
subject_first = {}
for subject in np.unique(groups):
    subject_mask = groups == subject
    subject_updrs = updrs_original[subject_mask]
    subject_updrs = subject_updrs[subject_updrs != 99]
    subject_first[subject] = subject_updrs[0] if len(subject_updrs) > 0 else 0

y_first = np.array([1 if subject_first[g] > 0 else 0 for g in groups])

print("\n📊 Comparación de etiquetas:")
print(f"  Original: {np.bincount(y_original)}")
print(f"  Modal:    {np.bincount(y_modal)}")
print(f"  Máximo:   {np.bincount(y_max)}")
print(f"  Primera:  {np.bincount(y_first)}")

# =============================================================================
# 4. EVALUAR CADA ESTRATEGIA
# =============================================================================
print("\n" + "="*60)
print("4. EVALUACIÓN DE ESTRATEGIAS")
print("="*60)

strategies = {
    'Original (por ventana)': y_original,
    'Modal (moda por sujeto)': y_modal,
    'Máximo (peor severidad)': y_max,
    'Primera sesión': y_first
}

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

results = []

for strategy_name, y_labels in strategies.items():
    print(f"\n🔬 Evaluando: {strategy_name}")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    auc_scores = []
    
    for train_idx, test_idx in sgkf.split(X_scaled, y_labels, groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        if len(np.unique(y_test)) < 2:
            continue
        
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, 
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        f1_scores.append(f1_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))
    
    results.append({
        'strategy': strategy_name,
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'n_folds': len(f1_scores)
    })
    
    print(f"  F1: {results[-1]['f1_mean']:.4f} ± {results[-1]['f1_std']:.4f}")
    print(f"  AUC: {results[-1]['auc_mean']:.4f} ± {results[-1]['auc_std']:.4f}")

# =============================================================================
# 5. RESULTADOS FINALES
# =============================================================================
print("\n" + "="*60)
print("5. COMPARATIVA FINAL")
print("="*60)

results_df = pd.DataFrame(results)
print(results_df.to_string())

# Figura comparativa
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.35

ax.bar(x - width/2, results_df['f1_mean'], width, yerr=results_df['f1_std'], 
       label='F1-score', capsize=5, color='blue', alpha=0.7)
ax.bar(x + width/2, results_df['auc_mean'], width, yerr=results_df['auc_std'], 
       label='AUC', capsize=5, color='green', alpha=0.7)

ax.set_xlabel('Estrategia')
ax.set_ylabel('Puntuación')
ax.set_title('Comparación de Estrategias para UPDRS Inconsistente')
ax.set_xticks(x)
ax.set_xticklabels(results_df['strategy'], rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/updrs_strategies_comparison.png', dpi=300)
plt.close()
print("\n✅ Figura guardada: analysis/updrs_strategies_comparison.png")

# =============================================================================
# 6. CONCLUSIÓN
# =============================================================================
print("\n" + "="*60)
print("6. CONCLUSIÓN Y RECOMENDACIÓN")
print("="*60)

best = results_df.loc[results_df['f1_mean'].idxmax()]
print(f"\n🏆 MEJOR ESTRATEGIA: {best['strategy']}")
print(f"   F1-score: {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
print(f"   AUC: {best['auc_mean']:.4f} ± {best['auc_std']:.4f}")

print("\n📝 RECOMENDACIÓN:")
if best['strategy'] == 'Original (por ventana)':
    print("  ✅ Usar etiquetas originales por ventana")
    print("     - Refleja variabilidad real del paciente")
    print("     - Mejor rendimiento predictivo")
    print("     - Aceptable para transfer learning")
elif best['strategy'] == 'Modal (moda por sujeto)':
    print("  ✅ Usar moda de UPDRS por sujeto")
    print("     - Consistente con la mayoría de evaluaciones")
    print("     - Reduce ruido en etiquetas")
else:
    print("  ✅ Usar estrategia basada en contexto clínico")

# Guardar resultados
results_df.to_csv('analysis/updrs_strategy_results.csv', index=False)
print("\n✅ Resultados guardados en analysis/updrs_strategy_results.csv")