# =============================================================================
# scripts/pca_domain_shift.py
# Generar figura PCA para cuantificar domain shift entre clínica y casa
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*60)
print("📊 GENERANDO PCA PARA DOMAIN SHIFT")
print("="*60)

# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print("\n📂 Cargando datos...")

loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Filtrar ejercicio 6
df_clinic = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()
df_home = df[(df['Contexto_sesion'] == 0) & (df['Ejercicio'] == 6)].copy()

print(f"  Clínica: {len(df_clinic):,} filas")
print(f"  Casa: {len(df_home):,} filas")

# =============================================================================
# 2. EXTRAER FEATURES (muestra representativa)
# =============================================================================
print("\n🔧 Extrayendo features (muestra)...")

WINDOW_SIZE = 100
STEP_SIZE = 100
preprocessor = IMUPreprocessor(fs=50)

def extract_features_sample(df_data, max_windows=500):
    """Extrae features de una muestra representativa"""
    X = []
    
    for session in df_data['Sesion'].unique():
        if len(X) >= max_windows:
            break
            
        df_ses = df_data[df_data['Sesion'] == session]
        acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < WINDOW_SIZE:
            continue
        
        for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
            if len(X) >= max_windows:
                break
                
            acc_window = acc[i:i+WINDOW_SIZE]
            gyro_window = gyro[i:i+WINDOW_SIZE]
            
            features = preprocessor.extract_features(acc_window, gyro_window)
            X.append(list(features.values()))
    
    return np.array(X)

# Extraer muestras
X_clinic_sample = extract_features_sample(df_clinic, max_windows=500)
X_home_sample = extract_features_sample(df_home, max_windows=500)

print(f"  Muestra clínica: {X_clinic_sample.shape}")
print(f"  Muestra casa: {X_home_sample.shape}")

# =============================================================================
# 3. PCA
# =============================================================================
print("\n📊 Aplicando PCA...")

# Combinar y escalar
X_combined = np.vstack([X_clinic_sample, X_home_sample])
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_combined)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Separar por dominio
n_clinic = len(X_clinic_sample)
X_pca_clinic = X_pca[:n_clinic]
X_pca_home = X_pca[n_clinic:]

print(f"  Varianza explicada PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  Varianza explicada PC2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"  Varianza total: {pca.explained_variance_ratio_.sum():.1%}")

# =============================================================================
# 4. FIGURA
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
ax.scatter(X_pca_clinic[:, 0], X_pca_clinic[:, 1], 
           c='blue', alpha=0.5, s=20, label='Clinical (supervised)')
ax.scatter(X_pca_home[:, 0], X_pca_home[:, 1], 
           c='red', alpha=0.5, s=20, label='Home (unsupervised)')

# Medias
mean_clinic = X_pca_clinic.mean(axis=0)
mean_home = X_pca_home.mean(axis=0)
ax.scatter(mean_clinic[0], mean_clinic[1], c='darkblue', s=200, marker='X', edgecolors='black', linewidths=2)
ax.scatter(mean_home[0], mean_home[1], c='darkred', s=200, marker='X', edgecolors='black', linewidths=2)

# Elipses de confianza (2 std)
from matplotlib.patches import Ellipse

def plot_ellipse(ax, points, color, n_std=2):
    cov = np.cov(points.T)
    mean = points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', linewidth=2, linestyle='--')
    ax.add_patch(ellipse)

plot_ellipse(ax, X_pca_clinic, 'blue', n_std=1.5)
plot_ellipse(ax, X_pca_home, 'red', n_std=1.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('Domain Shift: Clinical vs Home Feature Distributions', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/pca_domain_shift.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Figura guardada: figures/pca_domain_shift.png")

# =============================================================================
# 5. ESTADÍSTICAS ADICIONALES
# =============================================================================
from scipy import stats

# Test de Kolmogorov-Smirnov en PC1
ks_stat, ks_p = stats.ks_2samp(X_pca_clinic[:, 0], X_pca_home[:, 0])
print(f"\n📊 Kolmogorov-Smirnov test en PC1:")
print(f"   KS statistic: {ks_stat:.3f}")
print(f"   p-value: {ks_p:.4f}")
print(f"   {'✅ Domain shift significativo' if ks_p < 0.01 else '❌ No significativo'}")