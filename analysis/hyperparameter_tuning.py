# =============================================================================
# analysis/hyperparameter_tuning.py
# Búsqueda sistemática de hiperparámetros para Random Forest
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, make_scorer
import time
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 BÚSQUEDA SISTEMÁTICA DE HIPERPARÁMETROS")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

# Extraer features
WINDOW_SIZE = 100
STEP_SIZE = 100
preprocessor = IMUPreprocessor(fs=50)

X, y, groups = [], [], []

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
        
        updrs = df_ses['UPDRS'].iloc[0]
        label = 1 if updrs not in [0, 99] else 0
        y.append(label)
        groups.append(df_ses['subject_id'].iloc[0])

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

# Escalar
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n📊 Dataset: {X_scaled.shape[0]} ventanas, {X_scaled.shape[1]} features")
print(f"Clases: {np.bincount(y)}")

# =============================================================================
# GRID SEARCH DE HIPERPARÁMETROS
# =============================================================================
print("\n" + "="*60)
print("📊 GRID SEARCH - BÚSQUEDA SISTEMÁTICA")
print("="*60)

# Definir grid de parámetros
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', 0.3, 0.5]
}

# Crear scorer para F1
scorer = make_scorer(f1_score)

# StratifiedGroupKFold para validación
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search (limitado a pocas combinaciones para tiempo razonable)
# Nota: Esto puede tomar varios minutos
print("\n🔍 Probando combinaciones de hiperparámetros...")
print("(Esto puede tomar 5-10 minutos)")

rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=sgkf.split(X_scaled, y, groups),
    scoring=scorer,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search.fit(X_scaled, y)
elapsed = time.time() - start_time

print(f"\n✅ Búsqueda completada en {elapsed:.1f} segundos")

# =============================================================================
# RESULTADOS
# =============================================================================
print("\n" + "="*60)
print("📊 MEJORES HIPERPARÁMETROS ENCONTRADOS")
print("="*60)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\n🎯 Mejor F1-score: {best_score:.4f}")
print(f"\n📋 Mejores parámetros:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# =============================================================================
# TABLA COMPARATIVA DE COMBINACIONES
# =============================================================================
print("\n" + "="*60)
print("📊 TOP 10 COMBINACIONES")
print("="*60)

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)

top_results = results_df[['param_n_estimators', 'param_max_depth', 
                          'param_min_samples_split', 'param_min_samples_leaf',
                          'param_max_features', 'mean_test_score', 'std_test_score']].head(10)

print(top_results.to_string())

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
print("\n" + "="*60)
print("📊 VISUALIZACIÓN DE RESULTADOS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. n_estimators vs F1
ax1 = axes[0, 0]
n_est_results = results_df.groupby('param_n_estimators')['mean_test_score'].mean()
ax1.plot(n_est_results.index, n_est_results.values, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('F1-score medio')
ax1.set_title('Impacto de n_estimators')
ax1.grid(True, alpha=0.3)

# 2. max_depth vs F1
ax2 = axes[0, 1]
max_depth_results = results_df.groupby('param_max_depth')['mean_test_score'].mean()
max_depth_results.index = [str(x) for x in max_depth_results.index]
ax2.bar(max_depth_results.index, max_depth_results.values, alpha=0.7)
ax2.set_xlabel('max_depth')
ax2.set_ylabel('F1-score medio')
ax2.set_title('Impacto de max_depth')
ax2.grid(True, alpha=0.3)

# 3. Heatmap: n_estimators vs max_depth
ax3 = axes[1, 0]
pivot = results_df.pivot_table(index='param_n_estimators', 
                                columns='param_max_depth', 
                                values='mean_test_score')
im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=0.8)
ax3.set_xticks(range(len(pivot.columns)))
ax3.set_xticklabels(pivot.columns)
ax3.set_yticks(range(len(pivot.index)))
ax3.set_yticklabels(pivot.index)
ax3.set_xlabel('max_depth')
ax3.set_ylabel('n_estimators')
ax3.set_title('n_estimators vs max_depth')
plt.colorbar(im, ax=ax3, label='F1-score')

# 4. Resumen
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
RESUMEN DE OPTIMIZACIÓN:

Mejor F1-score: {best_score:.4f}

Parámetros óptimos:
• n_estimators: {best_params.get('n_estimators', '?')}
• max_depth: {best_params.get('max_depth', '?')}
• min_samples_split: {best_params.get('min_samples_split', '?')}
• min_samples_leaf: {best_params.get('min_samples_leaf', '?')}
• max_features: {best_params.get('max_features', '?')}

Comparación con parámetros anteriores:
• Antes (300,12,5,2): F1=0.789
• Ahora óptimo: F1={best_score:.4f}

Recomendación para reducir overfitting:
• Usar max_depth ≤ 8
• min_samples_split ≥ 5
• min_samples_leaf ≥ 2
"""
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('analysis/statistical/hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/statistical/hyperparameter_tuning.png")

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
top_results.to_csv('analysis/statistical/hyperparameter_results.csv', index=False)
print("\n✅ Resultados guardados: analysis/statistical/hyperparameter_results.csv")

# =============================================================================
# MODELO FINAL CON MEJORES PARÁMETROS
# =============================================================================
print("\n" + "="*60)
print("🎯 ENTRENANDO MODELO FINAL CON MEJORES PARÁMETROS")
print("="*60)

best_rf = RandomForestClassifier(
    n_estimators=best_params.get('n_estimators', 150),
    max_depth=best_params.get('max_depth', 8),
    min_samples_split=best_params.get('min_samples_split', 5),
    min_samples_leaf=best_params.get('min_samples_leaf', 2),
    max_features=best_params.get('max_features', 'sqrt'),
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Validación cruzada con mejores parámetros
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

for train_idx, test_idx in sgkf.split(X_scaled, y, groups):
    if len(np.unique(y[test_idx])) < 2:
        continue
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred))

print(f"\n📊 F1-score con parámetros óptimos: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

print("\n✅ Búsqueda de hiperparámetros completada")