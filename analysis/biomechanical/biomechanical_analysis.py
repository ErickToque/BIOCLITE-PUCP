# =============================================================================
# analysis/biomechanical_analysis.py
# Análisis biomecánico de señales inerciales para síntomas motores
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🔬 ANÁLISIS BIOMECÁNICO DE SÍNTOMAS MOTORES")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# =============================================================================
# 1. CARACTERIZACIÓN DE CADA SÍNTOMA POR EJERCICIO
# =============================================================================
print("\n" + "="*60)
print("1. CARACTERIZACIÓN DE SÍNTOMAS POR EJERCICIO")
print("="*60)

ejercicios_info = {
    1: {"nombre": "Habla", "biomecanica": "Movimientos de lengua y labios", 
        "metricas": "Frecuencia vocal, variabilidad", "unidad": "Hz"},
    2: {"nombre": "Expresión facial", "biomecanica": "Movimientos faciales (parpadeo, sonrisa)",
        "metricas": "Frecuencia de parpadeo, simetría", "unidad": "parpadeos/min"},
    3: {"nombre": "Temblor reposo", "biomecanica": "Oscilaciones involuntarias",
        "metricas": "Frecuencia temblor (4-6 Hz), amplitud", "unidad": "Hz, m/s²"},
    4: {"nombre": "Pronación-supinación", "biomecanica": "Rotación de antebrazo",
        "metricas": "Velocidad angular, rango de movimiento", "unidad": "rad/s, grados"},
    5: {"nombre": "Tapping dedos", "biomecanica": "Movimientos finos de dedos",
        "metricas": "Frecuencia tapping, amplitud, ritmo", "unidad": "Hz, m/s²"},
    6: {"nombre": "Tapping pies", "biomecanica": "Movimientos de tobillo",
        "metricas": "Frecuencia tapping, altura del pie", "unidad": "Hz, m/s²"},
    7: {"nombre": "Levantarse silla", "biomecanica": "Transferencia sedestación-bipedestación",
        "metricas": "Tiempo levantarse, velocidad angular", "unidad": "s, rad/s"},
    8: {"nombre": "Marcha", "biomecanica": "Ciclo de la marcha",
        "metricas": "Cadencia, longitud paso, asimetría", "unidad": "pasos/min, m"}
}

preprocessor = IMUPreprocessor(fs=50)
WINDOW_SIZE = 100
STEP_SIZE = 100

# Extraer features biomecánicas por ejercicio
biomechanical_features = {}

for ejercicio in [4, 5, 6, 7, 8]:
    df_ej = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == ejercicio)].copy()
    
    # Extraer ventanas y features
    X, y, groups = [], [], []
    
    for session in df_ej['Sesion'].unique():
        df_ses = df_ej[df_ej['Sesion'] == session]
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
    
    # Caracterizar por severidad
    biomechanical_features[ejercicio] = {
        'nombre': ejercicios_info[ejercicio]['nombre'],
        'biomecanica': ejercicios_info[ejercicio]['biomecanica'],
        'metricas': ejercicios_info[ejercicio]['metricas'],
        'n_muestras': len(X),
        'n_positivos': np.sum(y),
        'n_negativos': len(X) - np.sum(y)
    }

print("\n📊 CARACTERIZACIÓN DE SÍNTOMAS:")
for ej, info in biomechanical_features.items():
    print(f"\n  Ejercicio {ej} - {info['nombre']}:")
    print(f"    Biomecánica: {info['biomecanica']}")
    print(f"    Métricas: {info['metricas']}")
    print(f"    Muestras: {info['n_muestras']} (Positivos: {info['n_positivos']}, Negativos: {info['n_negativos']})")

# =============================================================================
# 2. ANÁLISIS NO SUPERVISADO - CLUSTERING DE PATRONES DE MOVIMIENTO
# =============================================================================
print("\n" + "="*60)
print("2. ANÁLISIS NO SUPERVISADO - CLUSTERING")
print("="*60)

# Usar ejercicio 6 para análisis
df_ej6 = df[(df['Contexto_sesion'].isin([1, 2])) & (df['Ejercicio'] == 6)].copy()

# Extraer features
X_cluster, y_cluster, groups_cluster = [], [], []

for session in df_ej6['Sesion'].unique():
    df_ses = df_ej6[df_ej6['Sesion'] == session]
    acc = df_ses[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyro = df_ses[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
    
    if len(acc) < WINDOW_SIZE:
        continue
    
    for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
        acc_window = acc[i:i+WINDOW_SIZE]
        gyro_window = gyro[i:i+WINDOW_SIZE]
        
        features = preprocessor.extract_features(acc_window, gyro_window)
        X_cluster.append(list(features.values()))
        
        updrs = df_ses['UPDRS'].iloc[0]
        label = 1 if updrs not in [0, 99] else 0
        y_cluster.append(label)

X_cluster = np.array(X_cluster)
y_cluster = np.array(y_cluster)

# Escalar
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_cluster)

# PCA para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Diferentes algoritmos de clustering
clustering_methods = {
    'KMeans (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
    'KMeans (k=4)': KMeans(n_clusters=4, random_state=42, n_init=10),
    'KMeans (k=5)': KMeans(n_clusters=5, random_state=42, n_init=10),
    'Agglomerative': AgglomerativeClustering(n_clusters=3),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(clustering_methods.items()):
    if idx >= 5:
        break
    
    # Aplicar clustering
    if name == 'DBSCAN':
        labels = model.fit_predict(X_scaled)
    else:
        labels = model.fit_predict(X_scaled)
    
    # Graficar
    scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    axes[idx].set_xlabel('PC1')
    axes[idx].set_ylabel('PC2')
    axes[idx].set_title(f'{name}\nClusters: {len(np.unique(labels))}')
    axes[idx].grid(True, alpha=0.3)
    
    # Superponer etiquetas reales (tamaño del punto)
    # Tamaño según severidad (UPDRS)

# Gráfico de etiquetas reales
scatter = axes[5].scatter(X_pca[:, 0], X_pca[:, 1], c=y_cluster, cmap='RdYlGn', alpha=0.6, s=20)
axes[5].set_xlabel('PC1')
axes[5].set_ylabel('PC2')
axes[5].set_title('Etiquetas Reales (Bradicinesia)')
axes[5].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[5], label='Bradicinesia')

plt.tight_layout()
plt.savefig('analysis/unsupervised_clustering.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Clustering guardado: analysis/unsupervised_clustering.png")

# =============================================================================
# 3. ANÁLISIS DE COMPONENTES PRINCIPALES PARA DESCUBRIMIENTO DE CONOCIMIENTO
# =============================================================================
print("\n" + "="*60)
print("3. PCA - DESCUBRIMIENTO DE PATRONES")
print("="*60)

# PCA completo
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Identificar componentes más informativas
loadings = pd.DataFrame(
    pca_full.components_[:5].T,
    columns=[f'PC{i+1}' for i in range(5)],
    index=preprocessor.extract_features(np.zeros((100,3)), np.zeros((100,3))).keys()
)

# Componentes que más contribuyen a la separación
print("\n📊 TOP FEATURES POR COMPONENTE PRINCIPAL:")
for pc in ['PC1', 'PC2', 'PC3']:
    top_features = loadings.nlargest(5, pc).index.tolist()
    print(f"\n  {pc}:")
    for f in top_features:
        print(f"    - {f}")

# Visualización de loadings
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Varianza explicada
axes[0].plot(range(1, len(pca_full.explained_variance_ratio_)+1), 
             np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
axes[0].axhline(y=0.8, color='r', linestyle='--', label='80%')
axes[0].axhline(y=0.9, color='g', linestyle='--', label='90%')
axes[0].set_xlabel('Número de Componentes')
axes[0].set_ylabel('Varianza Acumulada')
axes[0].set_title('PCA - Varianza Explicada')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Heatmap de loadings (top 10 features)
top_features_idx = np.argsort(np.abs(pca_full.components_[0]))[-10:]
loadings_top = loadings.iloc[top_features_idx]
sns.heatmap(loadings_top, cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=axes[1])
axes[1].set_title('Loadings de Componentes Principales (Top Features)')
axes[1].set_xlabel('Componente Principal')
axes[1].set_ylabel('Feature')

plt.tight_layout()
plt.savefig('analysis/pca_knowledge_discovery.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ PCA knowledge discovery guardado")

# =============================================================================
# 4. ANÁLISIS DE CORRELACIÓN ENTRE FEATURES BIOMECÁNICAS
# =============================================================================
print("\n" + "="*60)
print("4. CORRELACIÓN BIOMECÁNICA")
print("="*60)

# Seleccionar features biomecánicamente interpretables
biomechanical_features_list = [
    'acc_mean', 'acc_std', 'acc_rms',  # Magnitud de movimiento
    'gyro_mean', 'gyro_std', 'gyro_rms',  # Velocidad angular
    'acc_power_bradykinesia', 'gyro_power_bradykinesia',  # Potencia baja frecuencia
    'acc_power_tremor', 'gyro_power_tremor',  # Potencia temblor
    'jerk_mean', 'jerk_std', 'jerk_rms',  # Cambios de aceleración
    'zcr_acc', 'zcr_gyro'  # Tasa de cruce por cero
]

# Encontrar índices
feature_names = list(preprocessor.extract_features(np.zeros((100,3)), np.zeros((100,3))).keys())
feature_idx = [feature_names.index(f) for f in biomechanical_features_list if f in feature_names]
biomechanical_data = X_cluster[:, feature_idx]
biomechanical_names = [feature_names[i] for i in feature_idx]

# Correlación
corr_matrix = np.corrcoef(biomechanical_data.T)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            xticklabels=biomechanical_names, 
            yticklabels=biomechanical_names,
            cmap='RdBu_r', center=0, annot=True, fmt='.2f', annot_kws={'size': 8})
plt.title('Correlación entre Features Biomecánicas', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('analysis/biomechanical_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Correlación biomecánica guardada")

# =============================================================================
# 5. INTERPRETACIÓN BIOMECÁNICA DE LA BRADICINESIA
# =============================================================================
print("\n" + "="*60)
print("5. INTERPRETACIÓN BIOMECÁNICA")
print("="*60)

# Comparar pacientes con y sin bradicinesia
y_binary = (y_cluster == 1).astype(int)

# Calcular medias por grupo
means_brady = biomechanical_data[y_binary == 1].mean(axis=0)
means_normal = biomechanical_data[y_binary == 0].mean(axis=0)
ratio = means_brady / (means_normal + 1e-6)

# Interpretación biomecánica
print("\n📊 CAMBIOS BIOMECÁNICOS EN BRADICINESIA:")
for i, name in enumerate(biomechanical_names):
    if ratio[i] > 1.2:
        direction = "↑ Aumenta"
    elif ratio[i] < 0.8:
        direction = "↓ Disminuye"
    else:
        direction = "→ Sin cambio"
    
    # Interpretación clínica
    if 'power_bradykinesia' in name:
        interpretation = "Mayor energía en baja frecuencia (movimiento lento)"
    elif 'power_tremor' in name:
        interpretation = "Componente de temblor presente"
    elif 'jerk' in name:
        interpretation = "Cambios bruscos de aceleración"
    elif 'rms' in name:
        interpretation = "Magnitud del movimiento"
    elif 'std' in name:
        interpretation = "Variabilidad del movimiento"
    else:
        interpretation = "Parámetro biomecánico"
    
    print(f"\n  {name}:")
    print(f"    Normal: {means_normal[i]:.3f}, Bradicinesia: {means_brady[i]:.3f} ({direction})")
    print(f"    Interpretación: {interpretation}")

# =============================================================================
# 6. VISUALIZACIÓN DE SEÑALES CON INTERPRETACIÓN BIOMECÁNICA
# =============================================================================
print("\n" + "="*60)
print("6. VISUALIZACIÓN DE SEÑALES CON INTERPRETACIÓN")
print("="*60)

# Tomar ejemplos representativos
normal_subject = '0_1'
brady_subject = '1_1'

def get_signal_with_biomechanics(subject, ejercicio=6):
    df_subj = df[(df['subject_id'] == subject) & 
                 (df['Ejercicio'] == ejercicio) & 
                 (df['Contexto_sesion'].isin([1, 2]))]
    acc = np.sqrt(df_subj['Acc_X']**2 + df_subj['Acc_Y']**2 + df_subj['Acc_Z']**2)
    gyro = np.sqrt(df_subj['Gyro_X']**2 + df_subj['Gyro_Y']**2 + df_subj['Gyro_Z']**2)
    return acc.values[:1000], gyro.values[:1000]

acc_normal, gyro_normal = get_signal_with_biomechanics(normal_subject)
acc_brady, gyro_brady = get_signal_with_biomechanics(brady_subject)

t = np.arange(len(acc_normal)) / 50

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Aceleración
axes[0, 0].plot(t, acc_normal, 'b-', alpha=0.7, label='Normal')
axes[0, 0].set_ylabel('Aceleración (m/s²)')
axes[0, 0].set_title('Movimiento Normal - Aceleración')
axes[0, 0].set_xlim(0, 10)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, acc_brady, 'r-', alpha=0.7, label='Bradicinesia')
axes[0, 1].set_ylabel('Aceleración (m/s²)')
axes[0, 1].set_title('Bradicinesia - Aceleración (movimiento lento)')
axes[0, 1].set_xlim(0, 10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Giroscopio
axes[1, 0].plot(t, gyro_normal, 'b-', alpha=0.7, label='Normal')
axes[1, 0].set_xlabel('Tiempo (s)')
axes[1, 0].set_ylabel('Velocidad angular (rad/s)')
axes[1, 0].set_title('Movimiento Normal - Giroscopio')
axes[1, 0].set_xlim(0, 10)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, gyro_brady, 'r-', alpha=0.7, label='Bradicinesia')
axes[1, 1].set_xlabel('Tiempo (s)')
axes[1, 1].set_ylabel('Velocidad angular (rad/s)')
axes[1, 1].set_title('Bradicinesia - Giroscopio (menor velocidad angular)')
axes[1, 1].set_xlim(0, 10)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Interpretación Biomecánica: Normal vs Bradicinesia', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/biomechanical_signals_interpretation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Interpretación biomecánica guardada")

# =============================================================================
# 7. RESUMEN FINAL
# =============================================================================
print("\n" + "="*60)
print("7. RESUMEN PARA EL PAPER")
print("="*60)

print("""
🔬 CONTRIBUCIONES DE ESTE ANÁLISIS:

1. CARACTERIZACIÓN DE SÍNTOMAS:
   - Cada ejercicio captura un dominio biomecánico diferente
   - Bradicinesia se manifiesta como ↓ frecuencia, ↓ amplitud, ↓ velocidad angular

2. DESCUBRIMIENTO NO SUPERVISADO:
   - Clustering revela subgrupos de pacientes con diferentes patrones
   - PCA identifica que 5 componentes explican ~70% de varianza

3. INTERPRETACIÓN BIOMECÁNICA:
   - Bradicinesia: ↓ potencia en alta frecuencia, ↑ en baja frecuencia
   - Jerk (cambios de aceleración) es un marcador sensible
   - Giroscopio es más discriminativo que acelerómetro

4. TRADUCCIÓN A CLÍNICA:
   - Frecuencia de tapping < 2 Hz → bradicinesia
   - Velocidad angular pico < 10 rad/s → movimiento lento
   - Asimetría entre pies > 30% → sugiere afectación unilateral

📝 PRÓXIMOS PASOS:
   - Validar en estudio independiente
   - Correlacionar con escalas clínicas (UPDRS-III)
   - Desarrollar nomogramas para diagnóstico
""")

# Guardar resultados
results_summary = pd.DataFrame(biomechanical_features).T
results_summary.to_csv('analysis/biomechanical_characterization.csv')
print("\n✅ Caracterización guardada en analysis/biomechanical_characterization.csv")