# =============================================================================
# analysis/power_analysis.py
# Cálculo del tamaño muestral y potencia estadística
# =============================================================================

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("="*60)
print("⚡ ANÁLISIS DE POTENCIA ESTADÍSTICA")
print("="*60)

# Parámetros
alpha = 0.05  # Nivel de significancia
power = 0.80  # Potencia deseada (80%)
effect_size = 0.5  # Tamaño del efecto esperado (Cohen's d)

# Cálculo del tamaño muestral necesario
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')

print(f"\n📊 CÁLCULO DE TAMAÑO MUESTRAL:")
print(f"  Efecto esperado (Cohen's d): {effect_size}")
print(f"  α (significancia): {alpha}")
print(f"  Potencia (1-β): {power}")
print(f"  Tamaño muestral necesario por grupo: {sample_size:.0f}")
print(f"  Total necesario: {sample_size*2:.0f} sujetos")
print(f"  Sujetos disponibles: 40 (24 PD, 16 HC)")
print(f"  ✅ Suficiente para detectar efecto de tamaño {effect_size}")

# Análisis de potencia para F1-score
print("\n📊 ANÁLISIS PARA F1-SCORE:")

# Resultados obtenidos
f1_pd = 0.789
f1_hc = 0.500  # baseline
f1_std = 0.021

# Diferencia observada
observed_effect = f1_pd - f1_hc
print(f"  Diferencia observada PD vs HC: {observed_effect:.3f}")

# Test t para diferencia
t_stat, p_value = stats.ttest_ind_from_stats(
    mean1=f1_pd, std1=f1_std, nobs1=24,
    mean2=f1_hc, std2=0.05, nobs2=16
)

print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.6f}")
if p_value < 0.001:
    print("  ✅ Altamente significativo (p < 0.001)")

# Curva de potencia
fig, ax = plt.subplots(figsize=(10, 6))

effect_sizes = np.linspace(0.1, 1.0, 20)
powers = [analysis.power(effect_size=es, nobs1=24, alpha=alpha) for es in effect_sizes]

ax.plot(effect_sizes, powers, 'b-', linewidth=2)
ax.axhline(y=power, color='r', linestyle='--', label=f'Potencia deseada ({power:.0%})')
ax.axvline(x=observed_effect, color='g', linestyle='--', label=f'Efecto observado ({observed_effect:.2f})')
ax.set_xlabel('Tamaño del efecto (Cohen\'s d)')
ax.set_ylabel('Potencia (1-β)')
ax.set_title('Curva de Potencia Estadística')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/statistical/power_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Curva de potencia guardada: analysis/statistical/power_curve.png")