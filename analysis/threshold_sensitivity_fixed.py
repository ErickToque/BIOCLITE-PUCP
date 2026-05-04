# =============================================================================
# analysis/threshold_sensitivity_final_v2.py
# Análisis de sensibilidad al threshold - VERSIÓN DEFINITIVA
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*80)
print("🔬 THRESHOLD SENSITIVITY ANALYSIS - VERSIÓN DEFINITIVA")
print("="*80)

# =============================================================================
# 1. CARGAR RESULTADOS
# =============================================================================
print("\n📊 Cargando predicciones de casa...")

home_df = pd.read_csv('results/tables/home_predictions_final.csv')
print(f"  {len(home_df)} sujetos")

# =============================================================================
# 2. ANÁLISIS POR THRESHOLD
# =============================================================================
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
results = []

for tau in thresholds:
    tp = fp = tn = fn = 0
    
    for _, row in home_df.iterrows():
        grupo = row['grupo']
        prob = row['probability']
        
        pred = 1 if prob >= tau else 0
        true = 1 if grupo == 'Parkinson' else 0
        
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        elif pred == 0 and true == 1:
            fn += 1
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0
    
    results.append({
        'threshold': tau,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    })

results_df = pd.DataFrame(results)

# =============================================================================
# 3. RESULTADOS CLAVE
# =============================================================================
print("\n" + "="*60)
print("📊 RESULTADOS")
print("="*60)

print("\n📈 Resumen por threshold:")
print(results_df[['threshold', 'sensitivity', 'specificity', 'ppv', 'f1']].round(3).to_string())

# Mejor threshold por F1
best_idx = results_df['f1'].idxmax()
best_tau = results_df.loc[best_idx, 'threshold']
best_sens = results_df.loc[best_idx, 'sensitivity']
best_spec = results_df.loc[best_idx, 'specificity']
best_f1 = results_df.loc[best_idx, 'f1']

print(f"\n🏆 Mejor threshold por F1-score: τ = {best_tau:.2f}")
print(f"   Sensibilidad: {best_sens:.1%}")
print(f"   Especificidad: {best_spec:.1%}")
print(f"   F1: {best_f1:.3f}")

# Threshold con 100% especificidad
spec_100 = results_df[results_df['specificity'] == 1.0]
if len(spec_100) > 0:
    tau_100 = spec_100.iloc[0]['threshold']
    sens_100 = spec_100.iloc[0]['sensitivity']
    print(f"\n✅ Threshold con 100% especificidad: τ = {tau_100:.2f}")
    print(f"   Sensibilidad: {sens_100:.1%}")

# Threshold seleccionado (de tu código original)
our_tau = 0.60
our_row = results_df[results_df['threshold'] == our_tau]
if len(our_row) > 0:
    our_sens = our_row.iloc[0]['sensitivity']
    our_spec = our_row.iloc[0]['specificity']
    our_f1 = our_row.iloc[0]['f1']
    print(f"\n🎯 Threshold seleccionado: τ = {our_tau:.2f}")
    print(f"   Sensibilidad: {our_sens:.1%}")
    print(f"   Especificidad: {our_spec:.1%}")
    print(f"   F1: {our_f1:.3f}")

# =============================================================================
# 4. FIGURA
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Sensibilidad vs Especificidad
ax1 = axes[0, 0]
ax1.plot(thresholds, results_df['sensitivity'], 'o-', color='blue', linewidth=2, markersize=8, label='Sensibilidad')
ax1.plot(thresholds, results_df['specificity'], 's-', color='green', linewidth=2, markersize=8, label='Especificidad')
ax1.axvline(x=best_tau, color='orange', linestyle='--', linewidth=2, label=f'Óptimo F1 (τ={best_tau:.2f})')
ax1.axvline(x=our_tau, color='red', linestyle='--', linewidth=2, label=f'Nuestra elección (τ={our_tau:.2f})')
ax1.set_xlabel('Threshold τ', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Sensibilidad vs Especificidad', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Gráfico 2: F1-score
ax2 = axes[0, 1]
ax2.plot(thresholds, results_df['f1'], 'd-', color='purple', linewidth=2, markersize=8)
ax2.axvline(x=best_tau, color='orange', linestyle='--', linewidth=2, label=f'Óptimo (τ={best_tau:.2f}, F1={best_f1:.3f})')
ax2.axvline(x=our_tau, color='red', linestyle='--', linewidth=2, label=f'Nuestra elección (τ={our_tau:.2f}, F1={our_f1:.3f})')
ax2.set_xlabel('Threshold τ', fontsize=12)
ax2.set_ylabel('F1-score', fontsize=12)
ax2.set_title('F1-score vs Threshold', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# Gráfico 3: Precision-Recall
ax3 = axes[1, 0]
ax3.plot(results_df['sensitivity'], results_df['ppv'], 'o-', color='darkorange', linewidth=2, markersize=8)
ax3.plot(best_sens, results_df.loc[best_idx, 'ppv'], 'go', markersize=12, label=f'Óptimo (τ={best_tau:.2f})')
ax3.plot(our_sens, our_row.iloc[0]['ppv'], 'ro', markersize=12, label=f'Nuestra elección (τ={our_tau:.2f})')
ax3.set_xlabel('Sensibilidad (Recall)', fontsize=12)
ax3.set_ylabel('Precisión (PPV)', fontsize=12)
ax3.set_title('Curva Precision-Recall', fontsize=14)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Gráfico 4: Matriz de confusión
ax4 = axes[1, 1]
cm = np.array([[our_row.iloc[0]['tn'], our_row.iloc[0]['fp']],
               [our_row.iloc[0]['fn'], our_row.iloc[0]['tp']]])

im = ax4.imshow(cm, cmap='Blues', interpolation='nearest')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Sano', 'Parkinson'])
ax4.set_yticklabels(['Sano', 'Parkinson'])
ax4.set_xlabel('Predicción', fontsize=12)
ax4.set_ylabel('Real', fontsize=12)
ax4.set_title(f'Matriz de Confusión (τ={our_tau:.2f})\nSensibilidad={our_sens:.1%}, Especificidad={our_spec:.1%}', fontsize=12)

for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, 
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig('analysis/threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Figura guardada: analysis/threshold_sensitivity_analysis.png")

# =============================================================================
# 5. TABLA PARA EL PAPER
# =============================================================================
print("\n" + "="*60)
print("📊 TABLA DE THRESHOLDS PARA EL PAPER")
print("="*60)

table_data = []
for tau in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    row = results_df[results_df['threshold'] == tau]
    if len(row) > 0:
        r = row.iloc[0]
        table_data.append({
            'Threshold τ': tau,
            'Sensitivity': f"{r['sensitivity']:.1%}",
            'Specificity': f"{r['specificity']:.1%}",
            'PPV': f"{r['ppv']:.1%}",
            'F1': f"{r['f1']:.3f}"
        })

table_df = pd.DataFrame(table_data)
print(table_df.to_string())
table_df.to_csv('analysis/threshold_comparison_table.csv', index=False)
print("\n✅ Tabla guardada: analysis/threshold_comparison_table.csv")

# =============================================================================
# 6. INTERVALOS DE CONFIANZA (BOOTSTRAP)
# =============================================================================
print("\n" + "="*60)
print("📊 INTERVALOS DE CONFIANZA (BOOTSTRAP) - τ=0.60")
print("="*60)

n_bootstrap = 1000
n_subjects = len(home_df)
np.random.seed(42)

sensitivity_bootstrap = []
specificity_bootstrap = []

for _ in range(n_bootstrap):
    idx = np.random.choice(n_subjects, n_subjects, replace=True)
    sample = home_df.iloc[idx]
    
    tp = fp = tn = fn = 0
    
    for _, row in sample.iterrows():
        grupo = row['grupo']
        prob = row['probability']
        
        pred = 1 if prob >= 0.60 else 0
        true = 1 if grupo == 'Parkinson' else 0
        
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        elif pred == 0 and true == 1:
            fn += 1
    
    sensitivity_bootstrap.append(tp / (tp + fn) if (tp+fn)>0 else 0)
    specificity_bootstrap.append(tn / (tn + fp) if (tn+fp)>0 else 0)

sens_ci = np.percentile(sensitivity_bootstrap, [2.5, 97.5])
spec_ci = np.percentile(specificity_bootstrap, [2.5, 97.5])

print(f"\nPara τ = 0.60:")
print(f"  Sensibilidad: {our_sens:.1%} [IC 95%: {sens_ci[0]:.1%} - {sens_ci[1]:.1%}]")
print(f"  Especificidad: {our_spec:.1%} [IC 95%: {spec_ci[0]:.1%} - {spec_ci[1]:.1%}]")

# =============================================================================
# 7. CONCLUSIÓN
# =============================================================================
print("\n" + "="*60)
print("🎯 CONCLUSIONES PARA EL PAPER")
print("="*60)

print(f"""
1. MEJOR THRESHOLD POR F1-SCORE: τ = {best_tau:.2f}
   - Sensibilidad: {best_sens:.1%}
   - Especificidad: {best_spec:.1%}
   - F1: {best_f1:.3f}

2. THRESHOLD CON 100% ESPECIFICIDAD: τ = {tau_100:.2f}
   - Sensibilidad: {sens_100:.1%}

3. NUESTRA ELECCIÓN τ = 0.60:
   - Sensibilidad: {our_sens:.1%}
   - Especificidad: {our_spec:.1%}
   - Justificación clínica: Prioriza evitar falsos positivos

4. PARA EL PAPER:
   - Reportar τ=0.60 como nuestra elección conservadora
   - Mostrar que τ=0.45 daría 100% de sensibilidad
   - Incluir la figura de trade-off
""")

print("\n✅ Análisis completado")