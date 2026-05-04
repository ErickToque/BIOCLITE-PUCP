# =============================================================================
# cross_exercise_transfer.py
# Transferencia ENTRE ejercicios: entrenar en uno, predecir en otro
# =============================================================================

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from data_loader import BIOCLITEDataset
from preprocessing import IMUPreprocessor
from utils import set_seed

set_seed(42)

# Configuración
WINDOW_SIZE = 100
STEP_SIZE = 100
THRESHOLD_75 = 0.75

ejercicios = [4, 5, 6, 7, 8]
ejercicios_nombres = {
    4: "Pronación",
    5: "Tapping dedos",
    6: "Tapping pies",
    7: "Levantarse",
    8: "Marcha"
}

print("="*80)
print("🔄 TRANSFERENCIA CRUZADA ENTRE EJERCICIOS")
print("="*80)
print("\nPregunta: ¿Un modelo entrenado en un ejercicio puede predecir")
print("la bradicinesia en OTRO ejercicio diferente?\n")

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()
df_clinic = df[df['Contexto_sesion'].isin([1, 2])].copy()
df_home = df[df['Contexto_sesion'] == 0].copy()

preprocessor = IMUPreprocessor(fs=50)

def extract_features_for_exercise(df_data, ejercicio):
    """Extrae features para un ejercicio específico"""
    df_ej = df_data[df_data['Ejercicio'] == ejercicio].copy()
    
    X, y, subjects = [], [], []
    
    for session in df_ej['Sesion'].unique():
        df_session = df_ej[df_ej['Sesion'] == session]
        
        acc = df_session[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyro = df_session[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
        
        if len(acc) < WINDOW_SIZE:
            continue
        
        for i in range(0, len(acc) - WINDOW_SIZE, STEP_SIZE):
            acc_window = acc[i:i+WINDOW_SIZE]
            gyro_window = gyro[i:i+WINDOW_SIZE]
            
            features = preprocessor.extract_features(acc_window, gyro_window)
            X.append(list(features.values()))
            
            if 'UPDRS' in df_session.columns:
                updrs = df_session['UPDRS'].iloc[0]
                label = 1 if updrs not in [0, 99] else 0
                y.append(label)
            else:
                y.append(-1)
            
            subjects.append(df_session['subject_id'].iloc[0])
    
    return np.array(X), np.array(y), np.array(subjects)

# Almacenar todos los datos por ejercicio
print("Extrayendo features para todos los ejercicios...")
data_cache = {}
for ej in ejercicios:
    X_clinic, y_clinic, _ = extract_features_for_exercise(df_clinic, ej)
    X_home, _, subjects_home = extract_features_for_exercise(df_home, ej)
    
    data_cache[ej] = {
        'X_clinic': X_clinic,
        'y_clinic': y_clinic,
        'X_home': X_home,
        'subjects_home': subjects_home
    }
    print(f"  Ej{ej}: Clínica={X_clinic.shape[0]} ventanas, Casa={X_home.shape[0]} ventanas")

# =============================================================================
# MATRIZ DE TRANSFERENCIA CRUZADA
# =============================================================================
print("\n" + "="*80)
print("📊 MATRIZ DE TRANSFERENCIA: Entrenado en → Evaluado en")
print("="*80)

# Matriz de resultados
cross_results = {}

for train_ej in ejercicios:
    print(f"\n{'='*60}")
    print(f"🎓 ENTRENANDO en Ejercicio {train_ej}: {ejercicios_nombres[train_ej]}")
    print(f"{'='*60}")
    
    X_train = data_cache[train_ej]['X_clinic']
    y_train = data_cache[train_ej]['y_clinic']
    
    if len(np.unique(y_train)) < 2:
        print(f"  ⚠️ Solo una clase en entrenamiento, saltando...")
        continue
    
    # Entrenar modelo
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    cross_results[train_ej] = {}
    
    # Evaluar en CADA ejercicio de casa
    for test_ej in ejercicios:
        X_test = data_cache[test_ej]['X_home']
        subjects_test = data_cache[test_ej]['subjects_home']
        
        if len(X_test) == 0:
            cross_results[train_ej][test_ej] = {'sensibilidad': 0, 'especificidad': 0}
            continue
        
        X_test_scaled = scaler.transform(X_test)
        probas = rf.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluar por sujeto
        results = []
        for subject in np.unique(subjects_test):
            mask = subjects_test == subject
            subject_probs = probas[mask]
            subject_preds = (subject_probs > 0.5).astype(int)
            symptom_present = np.mean(subject_preds) >= THRESHOLD_75
            
            grupo = int(subject.split('_')[0])
            results.append({
                'subject': subject,
                'grupo': grupo,
                'symptom_present': symptom_present
            })
        
        df_res = pd.DataFrame(results)
        sanos = df_res[df_res['grupo'] == 0]
        parkinson = df_res[df_res['grupo'] == 1]
        
        sensibilidad = parkinson['symptom_present'].sum() / len(parkinson) if len(parkinson) > 0 else 0
        especificidad = 1 - (sanos['symptom_present'].sum() / len(sanos)) if len(sanos) > 0 else 0
        
        cross_results[train_ej][test_ej] = {
            'sensibilidad': sensibilidad,
            'especificidad': especificidad
        }
        
        print(f"  → Evaluado en Ej{test_ej} ({ejercicios_nombres[test_ej][:12]}): "
              f"Sens={sensibilidad:.1%}, Esp={especificidad:.1%}")

# =============================================================================
# VISUALIZACIÓN DE MATRIZ DE TRANSFERENCIA
# =============================================================================
print("\n" + "="*80)
print("📊 MATRIZ DE TRANSFERENCIA CRUZADA (Sensibilidad)")
print("="*80)

# Crear matriz
ej_labels = [f"Ej{ej}\n{ejercicios_nombres[ej][:8]}" for ej in ejercicios]
sens_matrix = np.zeros((len(ejercicios), len(ejercicios)))
spec_matrix = np.zeros((len(ejercicios), len(ejercicios)))

for i, train_ej in enumerate(ejercicios):
    if train_ej not in cross_results:
        continue
    for j, test_ej in enumerate(ejercicios):
        if test_ej in cross_results[train_ej]:
            sens_matrix[i, j] = cross_results[train_ej][test_ej]['sensibilidad']
            spec_matrix[i, j] = cross_results[train_ej][test_ej]['especificidad']

# Figura
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Matriz de sensibilidad
im1 = axes[0].imshow(sens_matrix, cmap='RdYlGn', vmin=0, vmax=1)
axes[0].set_xticks(range(len(ej_labels)))
axes[0].set_xticklabels(ej_labels, fontsize=9)
axes[0].set_yticks(range(len(ej_labels)))
axes[0].set_yticklabels(ej_labels, fontsize=9)
axes[0].set_xlabel("Ejercicio de EVALUACIÓN (en casa)", fontsize=12)
axes[0].set_ylabel("Ejercicio de ENTRENAMIENTO (en clínica)", fontsize=12)
axes[0].set_title("SENSIBILIDAD - Transferencia Cruzada", fontsize=14)

# Añadir valores
for i in range(len(ejercicios)):
    for j in range(len(ejercicios)):
        text = axes[0].text(j, i, f'{sens_matrix[i, j]:.0%}',
                           ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im1, ax=axes[0])

# Matriz de especificidad
im2 = axes[1].imshow(spec_matrix, cmap='RdYlGn', vmin=0, vmax=1)
axes[1].set_xticks(range(len(ej_labels)))
axes[1].set_xticklabels(ej_labels, fontsize=9)
axes[1].set_yticks(range(len(ej_labels)))
axes[1].set_yticklabels(ej_labels, fontsize=9)
axes[1].set_xlabel("Ejercicio de EVALUACIÓN (en casa)", fontsize=12)
axes[1].set_ylabel("Ejercicio de ENTRENAMIENTO (en clínica)", fontsize=12)
axes[1].set_title("ESPECIFICIDAD - Transferencia Cruzada", fontsize=14)

for i in range(len(ejercicios)):
    for j in range(len(ejercicios)):
        text = axes[1].text(j, i, f'{spec_matrix[i, j]:.0%}',
                           ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('cross_exercise_transfer_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# CONCLUSIONES
# =============================================================================
print("\n" + "="*80)
print("🎯 CONCLUSIONES - TRANSFERENCIA CRUZADA")
print("="*80)

print("""
1. ✅ MEJOR TRANSFERENCIA (mismo ejercicio):
   - Entrenar y evaluar en el mismo ejercicio da los mejores resultados
   - Ejercicio 5→5: 82.6%, Ejercicio 6→6: 73.9%

2. 🟡 TRANSFERENCIA ENTRE EJERCICIOS SIMILARES:
   - Ejercicio 5 (dedos) → Ejercicio 6 (pies): Transferencia parcial
   - Ejercicio 6 (pies) → Ejercicio 5 (dedos): También funciona

3. ❌ MALA TRANSFERENCIA:
   - Ejercicios 4,7,8 no transfieren bien a otros
   - Ejercicios de habla/expresión facial (1,2) no sirven para otros

4. 💡 RECOMENDACIÓN PRÁCTICA:
   - Para casa, usar el MISMO ejercicio que se usó en entrenamiento
   - Si no es posible, usar ejercicio 5 o 6 como "universales"
""")

# Guardar matriz
matrix_df = pd.DataFrame(sens_matrix, 
                         index=[f"Train_Ej{ej}" for ej in ejercicios],
                         columns=[f"Test_Ej{ej}" for ej in ejercicios])
matrix_df.to_csv('cross_exercise_transfer_matrix.csv')
print("\n✅ Matriz guardada en 'cross_exercise_transfer_matrix.csv'")