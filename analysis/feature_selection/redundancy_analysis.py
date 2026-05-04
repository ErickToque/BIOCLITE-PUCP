# =============================================================================
# analysis/feature_selection/redundancy_analysis.py - VERSIÓN CORREGIDA
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 ANÁLISIS DE REDUNDANCIA DE VARIABLES")
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

feature_names = list(preprocessor.extract_features(
    np.zeros((WINDOW_SIZE, 3)), 
    np.zeros((WINDOW_SIZE, 3))
).keys())

print(f"\n📊 Datos: {X.shape[0]} ventanas, {X.shape[1]} features")
print(f"Clases: Ausente={np.sum(y==0)}, Presente={np.sum(y==1)}")

# Escalar
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 1. CORRELACIÓN ENTRE FEATURES
# =============================================================================
print("\n" + "="*60)
print("1. MATRIZ DE CORRELACIÓN")
print("="*60)

corr_matrix = np.corrcoef(X_scaled.T)

# Detectar features altamente correlacionadas (>0.95)
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if abs(corr_matrix[i, j]) > 0.95:
            high_corr_pairs.append({
                'feature1': feature_names[i],
                'feature2': feature_names[j],
                'correlation': corr_matrix[i, j]
            })

print(f"Pares con correlación > 0.95: {len(high_corr_pairs)}")
if high_corr_pairs:
    print("\nTop 10 pares más correlacionados:")
    for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]:
        print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")

# Figura: heatmap de correlación
variances = np.var(X_scaled, axis=0)
top_var_idx = np.argsort(variances)[-20:]
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix[np.ix_(top_var_idx, top_var_idx)], 
            cmap='RdYlBu_r', center=0,
            xticklabels=[feature_names[i] for i in top_var_idx],
            yticklabels=[feature_names[i] for i in top_var_idx],
            cbar_kws={'label': 'Correlación'})
plt.title('Matriz de Correlación - Top 20 Features por Varianza', fontsize=14)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('analysis/feature_selection/correlation_matrix.png', dpi=300)
plt.close()
print("✅ Heatmap guardado")

# =============================================================================
# 2. PCA - ANÁLISIS DE DIMENSIONALIDAD
# =============================================================================
print("\n" + "="*60)
print("2. ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
print("="*60)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
n_components_90 = np.where(cumulative_variance >= 0.90)[0][0] + 1

print(f"Componentes para 90% varianza: {n_components_90}")
print(f"Componentes para 95% varianza: {n_components_95}")
print(f"Reducción dimensionalidad: {X.shape[1]} → {n_components_95} ({100*(1-n_components_95/X.shape[1]):.1f}% reducción)")

# Figura PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7)
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Varianza Explicada')
axes[0].set_title('Varianza Explicada por Componente')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-', linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
axes[1].axhline(y=0.90, color='g', linestyle='--', label='90% varianza')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Acumulada')
axes[1].set_title('Varianza Acumulada vs Componentes')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/feature_selection/pca_analysis.png', dpi=300)
plt.close()
print("✅ PCA analysis guardado")

# =============================================================================
# 3. MUTUAL INFORMATION
# =============================================================================
print("\n" + "="*60)
print("3. MUTUAL INFORMATION")
print("="*60)

mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\nTop 15 features por Mutual Information:")
print(mi_df.head(15).to_string())
mi_df.to_csv('analysis/feature_selection/mutual_information.csv', index=False)

# =============================================================================
# 4. RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*60)
print("4. RANDOM FOREST FEATURE IMPORTANCE")
print("="*60)

# Entrenar RF con validación cruzada para importancia robusta
'''# Modelo con overfitting
rf_final = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
'''
rf_final = RandomForestClassifier(n_estimators=150, max_depth=8,
                                   min_samples_split=8, min_samples_leaf=4,
                                   random_state=42, n_jobs=-1)
rf_final.fit(X_scaled, y)

rf_importance = rf_final.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_importance
}).sort_values('RF_Importance', ascending=False)

print("\nTop 15 features por Random Forest:")
print(rf_importance_df.head(15).to_string())

# =============================================================================
# 5. TABLA CONSOLIDADA
# =============================================================================
print("\n" + "="*60)
print("5. TABLA CONSOLIDADA DE IMPORTANCIA")
print("="*60)

consolidated_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_importance,
    'MI_Score': mi_scores,
    'Variance': variances,
    'Correlation_Redundant': [any(abs(corr_matrix[i, j]) > 0.95 for j in range(len(feature_names)) if j != i) for i in range(len(feature_names))]
}).sort_values('RF_Importance', ascending=False)

consolidated_df.to_csv('analysis/feature_selection/feature_importance_consolidated.csv', index=False)

# Mostrar top 20
print("\n📊 TOP 20 FEATURES (por Random Forest):")
print(consolidated_df.head(20)[['Feature', 'RF_Importance', 'MI_Score', 'Correlation_Redundant']].to_string())

# =============================================================================
# 6. FIGURA FINAL CONSOLIDADA
# =============================================================================
print("\n" + "="*60)
print("6. GENERANDO FIGURA FINAL")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Feature importance RF
top_20 = consolidated_df.head(20)
axes[0, 0].barh(range(len(top_20)), top_20['RF_Importance'].values[::-1])
axes[0, 0].set_yticks(range(len(top_20)))
axes[0, 0].set_yticklabels(top_20['Feature'].values[::-1], fontsize=8)
axes[0, 0].set_xlabel('Importancia')
axes[0, 0].set_title('Random Forest Feature Importance')
axes[0, 0].invert_yaxis()

# Mutual Information
top_20_mi = mi_df.head(20)
axes[0, 1].barh(range(len(top_20_mi)), top_20_mi['MI_Score'].values[::-1])
axes[0, 1].set_yticks(range(len(top_20_mi)))
axes[0, 1].set_yticklabels(top_20_mi['Feature'].values[::-1], fontsize=8)
axes[0, 1].set_xlabel('Mutual Information')
axes[0, 1].set_title('Mutual Information Scores')
axes[0, 1].invert_yaxis()

# PCA 2D
colors = ['blue' if label == 0 else 'red' for label in y]
axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].set_title('PCA 2D - Separación de clases')

# Varianza acumulada
axes[1, 1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-', linewidth=2)
axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95%')
axes[1, 1].axhline(y=0.90, color='g', linestyle='--', label='90%')
axes[1, 1].fill_between(range(1, len(cumulative_variance)+1), cumulative_variance, 0.95, 
                         where=cumulative_variance > 0.95, alpha=0.3, color='red')
axes[1, 1].set_xlabel('Componentes')
axes[1, 1].set_ylabel('Varianza acumulada')
axes[1, 1].set_title(f'PCA: {n_components_95} componentes para 95% varianza')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/feature_selection/consolidated_analysis.png', dpi=300)
plt.close()
print("✅ Figura consolidada guardada")

print("\n" + "="*60)
print("✅ ANÁLISIS DE REDUNDANCIA COMPLETADO")
print("="*60)
print("\nArchivos generados:")
print("  - analysis/feature_selection/correlation_matrix.png")
print("  - analysis/feature_selection/pca_analysis.png")
print("  - analysis/feature_selection/consolidated_analysis.png")
print("  - analysis/feature_selection/mutual_information.csv")
print("  - analysis/feature_selection/feature_importance_consolidated.csv")