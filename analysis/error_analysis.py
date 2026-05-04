# =============================================================================
# analysis/error_analysis_fixed.py
# Análisis de error corregido
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

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*60)
print("👥 ANÁLISIS DE ERROR POR PACIENTE (CORREGIDO)")
print("="*60)

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

# Entrenar modelo final
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=150, max_depth=8, 
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)

# Predicciones por sujeto
subject_results = []
for subject in np.unique(groups):
    mask = groups == subject
    X_subj = X_scaled[mask]
    y_subj = y[mask]
    
    y_pred = rf.predict(X_subj)
    y_prob = rf.predict_proba(X_subj)[:, 1]
    
    correct = (y_pred == y_subj).sum()
    total = len(y_subj)
    
    # Obtener UPDRS real del sujeto (manejar caso vacío)
    df_subj = df_clinic[df_clinic['subject_id'] == subject]
    updrs_values = [u for u in df_subj['UPDRS'].unique() if u != 99]
    updrs = updrs_values[0] if len(updrs_values) > 0 else 0
    
    subject_results.append({
        'subject': subject,
        'grupo': 'PD' if subject.startswith('1') else 'HC',
        'updrs': updrs,
        'n_windows': total,
        'correct': correct,
        'accuracy': correct / total,
        'mean_probability': np.mean(y_prob),
        'std_probability': np.std(y_prob)
    })

results_df = pd.DataFrame(subject_results)

# Identificar errores
results_df['error'] = 1 - results_df['accuracy']
results_df = results_df.sort_values('error', ascending=False)

print("\n📊 SUJETOS CON MAYOR TASA DE ERROR:")
print(results_df[results_df['error'] > 0].head(10).to_string())

# Figura
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Accuracy por sujeto
ax1 = axes[0, 0]
colors = ['red' if r['error'] > 0 else 'green' for _, r in results_df.iterrows()]
ax1.bar(range(len(results_df)), results_df['accuracy'], color=colors, alpha=0.7)
ax1.axhline(y=0.789, color='blue', linestyle='--', label='Media global (0.789)')
ax1.set_xlabel('Sujeto')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy por Sujeto')
ax1.legend()

# 2. Error vs UPDRS
ax2 = axes[0, 1]
pd_data = results_df[results_df['grupo'] == 'PD']
if len(pd_data) > 0:
    scatter = ax2.scatter(pd_data['updrs'], pd_data['error'], c=pd_data['error'], 
                          cmap='RdYlGn_r', s=100, alpha=0.7)
    ax2.set_xlabel('UPDRS')
    ax2.set_ylabel('Tasa de Error')
    ax2.set_title('Error vs Severidad (UPDRS)')
    plt.colorbar(scatter, ax=ax2)

# 3. Distribución de probabilidades
ax3 = axes[1, 0]
for _, row in results_df.iterrows():
    color = 'red' if row['error'] > 0.2 else 'green' if row['error'] < 0.1 else 'orange'
    ax3.scatter(row['mean_probability'], row['std_probability'], 
                c=color, s=100, alpha=0.7, edgecolors='black')
    ax3.annotate(row['subject'], (row['mean_probability'], row['std_probability']), fontsize=8)

ax3.set_xlabel('Probabilidad media')
ax3.set_ylabel('Desviación estándar de probabilidad')
ax3.set_title('Confianza del Modelo por Sujeto')

# 4. Resumen
ax4 = axes[1, 1]
ax4.axis('off')
error_summary = f"""
RESUMEN DE ERRORES:

Sujetos correctamente clasificados: {(results_df['error'] == 0).sum()}/{len(results_df)}
Sujetos con error > 0: {(results_df['error'] > 0).sum()}

Pacientes PD mal clasificados:
{results_df[(results_df['grupo'] == 'PD') & (results_df['error'] > 0)]['subject'].tolist()}

Posibles causas:
• UPDRS bajo (síntomas leves)
• Variabilidad intra-paciente
• Datos insuficientes en ciertos sujetos
• Posible overfitting detectado (brecha 0.21)
"""
ax4.text(0.05, 0.95, error_summary, transform=ax4.transAxes, fontsize=9, 
         verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('analysis/patient_analysis/error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Análisis de error guardado: analysis/patient_analysis/error_analysis.png")

# Guardar resultados
results_df.to_csv('analysis/patient_analysis/patient_results.csv', index=False)
print("✅ Resultados guardados: analysis/patient_analysis/patient_results.csv")

# Mostrar resumen
print("\n📊 RESUMEN DE ERRORES POR SUJETO:")
print(results_df[['subject', 'grupo', 'updrs', 'accuracy', 'error']].to_string())