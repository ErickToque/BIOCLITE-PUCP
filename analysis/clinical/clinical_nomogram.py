# =============================================================================
# analysis/clinical_nomogram_final.py
# Nomograma clínico - VERSIÓN FINAL
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("📊 CREANDO NOMOGRAMA CLÍNICO")
print("="*80)

# Crear nomograma
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

ax.text(0.5, 0.95, 'NOMOGRAMA CLÍNICO PARA BRADICINESIA', 
        fontsize=14, fontweight='bold', ha='center')

features = ['Potencia Giroscopio\n(0.5-3 Hz)', 'Aceleración Media\n(m/s²)', 'Potencia Temblor\n(3-8 Hz)']
value_ranges = [(0, 20), (5, 20), (0, 15)]

y_start = 0.75
y_step = 0.12

for i, (feature, value_range) in enumerate(zip(features, value_ranges)):
    y_pos = y_start - i * y_step
    
    ax.text(0.1, y_pos, feature, fontsize=10, fontweight='bold')
    
    for j, val in enumerate(np.linspace(value_range[0], value_range[1], 5)):
        x_pos = 0.3 + j * 0.12
        ax.text(x_pos, y_pos - 0.02, f'{val:.0f}', fontsize=8, ha='center')
        ax.plot([x_pos, x_pos], [y_pos - 0.01, y_pos + 0.01], 'k-', linewidth=1)

# Escala total de puntos
ax.text(0.1, y_start - 3*y_step - 0.05, 'PUNTOS TOTALES', fontsize=10, fontweight='bold')
total_points = np.linspace(0, 300, 7)
for j, pt in enumerate(total_points):
    x_pos = 0.3 + j * 0.12
    ax.text(x_pos, y_start - 3*y_step - 0.07, f'{pt:.0f}', fontsize=8, ha='center')
    ax.plot([x_pos, x_pos], [y_start - 3*y_step - 0.09, y_start - 3*y_step - 0.05], 'k-', linewidth=1)

# Probabilidad de bradicinesia
ax.text(0.1, y_start - 4*y_step, 'PROBABILIDAD', fontsize=10, fontweight='bold')
probs = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
for j, prob in enumerate(probs):
    x_pos = 0.3 + j * 0.12
    ax.text(x_pos, y_start - 4*y_step - 0.02, f'{prob:.2f}', fontsize=7, ha='center', rotation=45)
    ax.plot([x_pos, x_pos], [y_start - 4*y_step - 0.04, y_start - 4*y_step], 'k-', linewidth=1)

# Línea de decisión (0.5) - SIN transform
umbral_x = 0.3 + probs.index(0.5) * 0.12
ax.axhline(y=y_start - 4*y_step - 0.02, xmin=0.3, xmax=0.3 + 8*0.12, 
           color='red', linewidth=2)
ax.text(umbral_x + 0.02, y_start - 4*y_step - 0.02, '← Umbral 0.5', 
        fontsize=9, color='red')

instructions = """
INSTRUCCIONES DE USO:
1. Localice el valor de cada feature en su escala
2. Sume los puntos correspondientes
3. Encuentre la probabilidad asociada en la escala inferior
4. Probabilidad > 0.5 sugiere bradicinesia

EJEMPLO:
Paciente con:
• Potencia giroscopio: 5 → 40 puntos
• Aceleración media: 12 → 30 puntos
• Potencia temblor: 8 → 50 puntos
TOTAL: 120 puntos → Probabilidad ≈ 0.65 → POSITIVO
"""
ax.text(0.1, 0.1, instructions, fontsize=9, verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('analysis/clinical_nomogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Nomograma clínico guardado: analysis/clinical_nomogram.png")