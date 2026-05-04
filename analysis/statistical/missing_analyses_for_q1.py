# =============================================================================
# missing_analyses_for_q1.py
# Análisis faltantes para publicación de alto impacto
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📋 CHECKLIST PARA PUBLICACIÓN Q1")
print("="*80)

checklist = {
    "1. Análisis Estadístico Riguroso": {
        "Pruebas de significancia": "❌",
        "Corrección de comparaciones múltiples": "❌",
        "Tamaño del efecto (Cohen's d)": "❌",
        "Intervalos de confianza (bootstrapping)": "❌"
    },
    "2. Validación Robusta": {
        "Validación externa (dataset independiente)": "❌",
        "Cross-validation anidada": "❌",
        "Análisis de sensibilidad": "❌"
    },
    "3. Reproducibilidad": {
        "Docker/Singularity container": "❌",
        "Repositorio público estructurado": "🟡",
        "Código documentado": "🟡"
    },
    "4. Interpretabilidad Clínica": {
        "SHAP values (completado)": "✅",
        "Nomogramas clínicos": "❌",
        "Reglas de decisión simples": "❌",
        "Correlación con escalas clínicas": "❌"
    },
    "5. Benchmarking": {
        "Comparación con SOTA": "🟡",
        "Ablation studies": "❌",
        "Análisis de complejidad": "❌"
    }
}

for category, items in checklist.items():
    print(f"\n{category}:")
    for item, status in items.items():
        print(f"  {status} {item}")