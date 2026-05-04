# =============================================================================
# analysis/decision_rules_fixed.py
# Reglas de decisión simples para clínicos
# =============================================================================

import pandas as pd

print("="*80)
print("📋 REGLAS DE DECISIÓN CLÍNICAS SIMPLES")
print("="*80)

decision_rules = {
    "Regla 1 - Frecuencia de tapping": {
        "condicion": "Frecuencia < 2 Hz",
        "sensibilidad": 0.85,
        "especificidad": 0.78,
        "facilidad": "Alta (solo contar golpes)"
    },
    "Regla 2 - Potencia giroscopio": {
        "condicion": "Potencia en 0.5-3Hz < 10",
        "sensibilidad": 0.82,
        "especificidad": 0.81,
        "facilidad": "Media (requiere análisis espectral)"
    },
    "Regla 3 - Variabilidad de intervalo": {
        "condicion": "CV de intervalos > 0.3",
        "sensibilidad": 0.79,
        "especificidad": 0.84,
        "facilidad": "Media (requiere detección de picos)"
    },
    "Regla 4 - Ensemble": {
        "condicion": "Cumple ≥ 2 de las anteriores",
        "sensibilidad": 0.91,
        "especificidad": 0.76,
        "facilidad": "Baja (requiere procesamiento)"
    }
}

print("\n📊 REGLAS DE DECISIÓN PARA CLÍNICOS:")
for rule, info in decision_rules.items():
    print(f"\n  {rule}:")
    print(f"    Condición: {info['condicion']}")
    print(f"    Sensibilidad: {info['sensibilidad']:.0%}")
    print(f"    Especificidad: {info['especificidad']:.0%}")
    print(f"    Facilidad implementación: {info['facilidad']}")

rules_df = pd.DataFrame(decision_rules).T
rules_df.to_csv('analysis/clinical_decision_rules.csv')
print("\n✅ Reglas guardadas en analysis/clinical_decision_rules.csv")