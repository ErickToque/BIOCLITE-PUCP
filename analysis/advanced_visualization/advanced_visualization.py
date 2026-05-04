# =============================================================================
# analysis/advanced_visualization.py
# Visualización avanzada: series de tiempo, ventanas, espectrogramas, features
# =============================================================================

import sys
sys.path.insert(0, '/home/etoque/BIOCLITE-PUCP')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, welch, find_peaks
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import BIOCLITEDataset
from src.data.preprocessor import IMUPreprocessor

print("="*80)
print("🎯 VISUALIZACIÓN AVANZADA: BRADICINESIA VS NORMAL")
print("="*80)

# Cargar datos
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Seleccionar sujetos representativos
normal_subject = '0_1'  # Sano
brady_subject = '1_1'   # Parkinson con bradicinesia

# Configuración
FS = 50
WINDOW_SIZE = 100  # 2 segundos
STEP_SIZE = 50     # 1 segundo overlap

# =============================================================================
# 1. SERIE DE TIEMPO COMPLETA CON MARCACIÓN DE SÍNTOMAS
# =============================================================================
print("\n" + "="*60)
print("1. SERIE DE TIEMPO CON MARCACIÓN DE SÍNTOMAS")
print("="*60)

def get_full_signal(subject, ejercicio=6):
    df_subj = df[(df['subject_id'] == subject) & 
                 (df['Ejercicio'] == ejercicio) & 
                 (df['Contexto_sesion'].isin([1, 2]))]
    acc = np.sqrt(df_subj['Acc_X']**2 + df_subj['Acc_Y']**2 + df_subj['Acc_Z']**2)
    gyro = np.sqrt(df_subj['Gyro_X']**2 + df_subj['Gyro_Y']**2 + df_subj['Gyro_Z']**2)
    updrs = df_subj['UPDRS'].values[0] if len(df_subj) > 0 else 0
    return acc.values, gyro.values, updrs

acc_normal, gyro_normal, updrs_normal = get_full_signal(normal_subject)
acc_brady, gyro_brady, updrs_brady = get_full_signal(brady_subject)

# Limitar a 30 segundos para visualización
duration = 30 * FS  # 30 segundos
acc_normal = acc_normal[:duration]
acc_brady = acc_brady[:duration]
gyro_normal = gyro_normal[:duration]
gyro_brady = gyro_brady[:duration]

t = np.arange(len(acc_normal)) / FS

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Sujeto Normal - Aceleración
axes[0].plot(t, acc_normal, 'b-', linewidth=1, alpha=0.8)
axes[0].set_ylabel('Aceleración (m/s²)', fontsize=10)
axes[0].set_title(f'Sujeto Normal (UPDRS={updrs_normal}) - Aceleración', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, 30)
axes[0].set_ylim(0, 25)
axes[0].grid(True, alpha=0.3)

# Marcar picos (golpes de pie)
peaks_normal, _ = find_peaks(acc_normal, distance=25, height=np.std(acc_normal))
axes[0].plot(t[peaks_normal], acc_normal[peaks_normal], 'ro', markersize=3, label=f'Golpes (n={len(peaks_normal)})')
axes[0].legend()

# Sujeto Bradicinesia - Aceleración
axes[1].plot(t, acc_brady, 'r-', linewidth=1, alpha=0.8)
axes[1].set_ylabel('Aceleración (m/s²)', fontsize=10)
axes[1].set_title(f'Bradicinesia (UPDRS={updrs_brady}) - Aceleración', fontsize=12, fontweight='bold')
axes[1].set_xlim(0, 30)
axes[1].set_ylim(0, 25)
axes[1].grid(True, alpha=0.3)

peaks_brady, _ = find_peaks(acc_brady, distance=25, height=np.std(acc_brady))
axes[1].plot(t[peaks_brady], acc_brady[peaks_brady], 'go', markersize=3, label=f'Golpes (n={len(peaks_brady)})')
axes[1].legend()

# Sujeto Normal - Giroscopio
axes[2].plot(t, gyro_normal, 'b-', linewidth=1, alpha=0.8)
axes[2].set_ylabel('Velocidad angular (rad/s)', fontsize=10)
axes[2].set_title(f'Sujeto Normal - Giroscopio', fontsize=12)
axes[2].set_xlim(0, 30)
axes[2].grid(True, alpha=0.3)

# Sujeto Bradicinesia - Giroscopio
axes[3].plot(t, gyro_brady, 'r-', linewidth=1, alpha=0.8)
axes[3].set_ylabel('Velocidad angular (rad/s)', fontsize=10)
axes[3].set_xlabel('Tiempo (segundos)', fontsize=10)
axes[3].set_title(f'Bradicinesia - Giroscopio', fontsize=12)
axes[3].set_xlim(0, 30)
axes[3].grid(True, alpha=0.3)

plt.suptitle('Comparación de Señales: Normal vs Bradicinesia (Tapping de pies)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/time_series_with_symptom_markers.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/time_series_with_symptom_markers.png")

# =============================================================================
# 2. VENTANAS DE TIEMPO - PATRONES DE SÍNTOMA
# =============================================================================
print("\n" + "="*60)
print("2. VENTANAS DE TIEMPO - PATRONES DE SÍNTOMA")
print("="*60)

# Extraer ventanas individuales
def extract_windows(signal, window_size=100, step=50):
    windows = []
    for i in range(0, len(signal) - window_size, step):
        windows.append(signal[i:i+window_size])
    return np.array(windows)

windows_normal = extract_windows(acc_normal, WINDOW_SIZE, STEP_SIZE)
windows_brady = extract_windows(acc_brady, WINDOW_SIZE, STEP_SIZE)

# Seleccionar ventanas representativas
t_windows = np.arange(WINDOW_SIZE) / FS

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Ventanas normales
for i in range(3):
    axes[0, i].plot(t_windows, windows_normal[i], 'b-', linewidth=1.5)
    axes[0, i].set_title(f'Ventana Normal {i+1}', fontsize=10)
    axes[0, i].set_ylabel('Aceleración (m/s²)')
    axes[0, i].set_xlabel('Tiempo (s)')
    axes[0, i].set_ylim(0, 25)
    axes[0, i].grid(True, alpha=0.3)
    
    # Marcar picos en ventana
    peaks, _ = find_peaks(windows_normal[i], distance=15)
    axes[0, i].plot(t_windows[peaks], windows_normal[i][peaks], 'ro', markersize=4)

# Ventanas con bradicinesia
for i in range(3):
    axes[1, i].plot(t_windows, windows_brady[i], 'r-', linewidth=1.5)
    axes[1, i].set_title(f'Ventana Bradicinesia {i+1}', fontsize=10)
    axes[1, i].set_ylabel('Aceleración (m/s²)')
    axes[1, i].set_xlabel('Tiempo (s)')
    axes[1, i].set_ylim(0, 25)
    axes[1, i].grid(True, alpha=0.3)
    
    # Marcar picos en ventana
    peaks, _ = find_peaks(windows_brady[i], distance=15)
    axes[1, i].plot(t_windows[peaks], windows_brady[i][peaks], 'go', markersize=4)

plt.suptitle('Patrones de Movimiento: Ventanas de 2 segundos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/symptom_pattern_windows.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/symptom_pattern_windows.png")

# =============================================================================
# 3. DOMINIO DE FRECUENCIA - ANÁLISIS ESPECTRAL
# =============================================================================
print("\n" + "="*60)
print("3. DOMINIO DE FRECUENCIA - ANÁLISIS ESPECTRAL")
print("="*60)

# Calcular PSD para cada tipo
f_normal, Pxx_normal = welch(acc_normal, fs=FS, nperseg=min(256, len(acc_normal)))
f_brady, Pxx_brady = welch(acc_brady, fs=FS, nperseg=min(256, len(acc_brady)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PSD comparativa
axes[0].semilogy(f_normal, Pxx_normal, 'b-', linewidth=2, label='Normal')
axes[0].semilogy(f_brady, Pxx_brady, 'r-', linewidth=2, label='Bradicinesia')
axes[0].set_xlim(0, 12)
axes[0].set_xlabel('Frecuencia (Hz)', fontsize=12)
axes[0].set_ylabel('Densidad Espectral de Potencia (g²/Hz)', fontsize=12)
axes[0].set_title('Densidad Espectral de Potencia', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bandas de interés
bands = {
    'Bradicinesia\n(0.5-3 Hz)': (0.5, 3),
    'Temblor\n(3-8 Hz)': (3, 8),
    'Alta Frecuencia\n(8-12 Hz)': (8, 12)
}

x_pos = np.arange(len(bands))
width = 0.35
normal_power = []
brady_power = []

for (band_name, (f_low, f_high)) in bands.items():
    mask = (f_normal >= f_low) & (f_normal <= f_high)
    normal_power.append(np.sum(Pxx_normal[mask]))
    brady_power.append(np.sum(Pxx_brady[mask]))

axes[1].bar(x_pos - width/2, normal_power, width, label='Normal', alpha=0.8, color='blue')
axes[1].bar(x_pos + width/2, brady_power, width, label='Bradicinesia', alpha=0.8, color='red')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(bands.keys(), fontsize=10)
axes[1].set_ylabel('Potencia Total', fontsize=12)
axes[1].set_title('Potencia por Banda Frecuencial', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Análisis en Dominio de Frecuencia', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/frequency_domain_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/frequency_domain_analysis.png")

# =============================================================================
# 4. FEATURES MÁS IMPORTANTES VISUALIZADAS
# =============================================================================
print("\n" + "="*60)
print("4. FEATURES MÁS IMPORTANTES")
print("="*60)

# Extraer features de ventanas individuales
preprocessor = IMUPreprocessor(fs=50)

features_normal = []
features_brady = []

for window in windows_normal[:50]:
    acc_window = np.column_stack([window, window, window])  # Simular 3 ejes
    gyro_window = np.zeros_like(acc_window)
    feats = preprocessor.extract_features(acc_window, gyro_window)
    features_normal.append(list(feats.values()))

for window in windows_brady[:50]:
    acc_window = np.column_stack([window, window, window])
    gyro_window = np.zeros_like(acc_window)
    feats = preprocessor.extract_features(acc_window, gyro_window)
    features_brady.append(list(feats.values()))

features_normal = np.array(features_normal)
features_brady = np.array(features_brady)

feature_names = list(feats.keys())

# Seleccionar top features (por diferencia de medias)
mean_normal = features_normal.mean(axis=0)
mean_brady = features_brady.mean(axis=0)
diff = np.abs(mean_normal - mean_brady)
top_indices = np.argsort(diff)[-8:][::-1]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, feat_idx in enumerate(top_indices):
    ax = axes[idx]
    data_normal = features_normal[:, feat_idx]
    data_brady = features_brady[:, feat_idx]
    
    # Boxplot
    bp = ax.boxplot([data_normal, data_brady], labels=['Normal', 'Bradicinesia'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax.set_title(feature_names[feat_idx].replace('_', ' '), fontsize=9)
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3)

plt.suptitle('Top 8 Features que Diferencian Normal de Bradicinesia', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/top_features_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/top_features_boxplots.png")

# =============================================================================
# 5. ESPECTROGRAMAS - ANÁLISIS TIEMPO-FRECUENCIA
# =============================================================================
print("\n" + "="*60)
print("5. ESPECTROGRAMAS - ANÁLISIS TIEMPO-FRECUENCIA")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Espectrograma - Normal
f, t_spec, Sxx = spectrogram(acc_normal, fs=FS, nperseg=128, noverlap=64)
im1 = axes[0, 0].pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
axes[0, 0].set_ylabel('Frecuencia (Hz)', fontsize=12)
axes[0, 0].set_title('Espectrograma - Normal', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(0, 12)
plt.colorbar(im1, ax=axes[0, 0], label='PSD (dB)')

# Espectrograma - Bradicinesia
f, t_spec, Sxx = spectrogram(acc_brady, fs=FS, nperseg=128, noverlap=64)
im2 = axes[0, 1].pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
axes[0, 1].set_ylabel('Frecuencia (Hz)', fontsize=12)
axes[0, 1].set_title('Espectrograma - Bradicinesia', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(0, 12)
plt.colorbar(im2, ax=axes[0, 1], label='PSD (dB)')

# Espectrograma de potencia relativa - Normal
axes[1, 0].plot(t_spec, np.mean(Sxx, axis=0), 'b-', linewidth=2)
axes[1, 0].set_xlabel('Tiempo (s)', fontsize=12)
axes[1, 0].set_ylabel('Potencia media (dB)', fontsize=12)
axes[1, 0].set_title('Evolución Temporal de Potencia - Normal', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Espectrograma de potencia relativa - Bradicinesia
axes[1, 1].plot(t_spec, np.mean(Sxx, axis=0), 'r-', linewidth=2)
axes[1, 1].set_xlabel('Tiempo (s)', fontsize=12)
axes[1, 1].set_ylabel('Potencia media (dB)', fontsize=12)
axes[1, 1].set_title('Evolución Temporal de Potencia - Bradicinesia', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Análisis Tiempo-Frecuencia (Espectrogramas)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/spectrogram_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/spectrogram_analysis.png")

# =============================================================================
# 6. ANÁLISIS DE RITMO - INTERVALOS ENTRE GOLPES
# =============================================================================
print("\n" + "="*60)
print("6. ANÁLISIS DE RITMO - VARIABILIDAD TEMPORAL")
print("="*60)

# Detectar picos en señal completa
peaks_normal, _ = find_peaks(acc_normal, distance=25, height=np.std(acc_normal))
peaks_brady, _ = find_peaks(acc_brady, distance=25, height=np.std(acc_brady))

# Calcular intervalos entre golpes
intervals_normal = np.diff(peaks_normal) / FS
intervals_brady = np.diff(peaks_brady) / FS

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución de intervalos
axes[0].hist(intervals_normal, bins=20, alpha=0.7, color='blue', label='Normal', density=True)
axes[0].hist(intervals_brady, bins=20, alpha=0.7, color='red', label='Bradicinesia', density=True)
axes[0].set_xlabel('Intervalo entre golpes (segundos)', fontsize=12)
axes[0].set_ylabel('Densidad', fontsize=12)
axes[0].set_title('Distribución de Intervalos de Tapping', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Evolución temporal de intervalos
axes[1].plot(peaks_normal[1:]/FS, intervals_normal, 'bo-', markersize=4, linewidth=1, alpha=0.7, label='Normal')
axes[1].plot(peaks_brady[1:]/FS, intervals_brady, 'ro-', markersize=4, linewidth=1, alpha=0.7, label='Bradicinesia')
axes[1].set_xlabel('Tiempo (s)', fontsize=12)
axes[1].set_ylabel('Intervalo (s)', fontsize=12)
axes[1].set_title('Evolución Temporal de Intervalos', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Análisis de Ritmo: Bradicinesia → Intervalos más largos y variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/rhythm_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura guardada: analysis/rhythm_analysis.png")

# =============================================================================
# 7. RESUMEN VISUAL COMPLETO
# =============================================================================
print("\n" + "="*60)
print("7. RESUMEN VISUAL - PÁGINA DE PAPER")
print("="*60)

# Crear figura compuesta para el paper
fig = plt.figure(figsize=(20, 24))

# Añadir título principal
fig.suptitle('Caracterización de Bradicinesia: Análisis Multidimensional', 
             fontsize=16, fontweight='bold', y=0.98)

# Subplot 1: Serie temporal con marcadores
ax1 = plt.subplot(4, 3, 1)
ax1.plot(t[:500], acc_normal[:500], 'b-', linewidth=1)
peaks_n, _ = find_peaks(acc_normal[:500], distance=25)
ax1.plot(t[peaks_n], acc_normal[peaks_n], 'ro', markersize=4)
ax1.set_ylabel('Aceleración (m/s²)')
ax1.set_title('Normal - Serie Temporal')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(4, 3, 2)
ax2.plot(t[:500], acc_brady[:500], 'r-', linewidth=1)
peaks_b, _ = find_peaks(acc_brady[:500], distance=25)
ax2.plot(t[peaks_b], acc_brady[peaks_b], 'go', markersize=4)
ax2.set_title('Bradicinesia - Serie Temporal')
ax2.grid(True, alpha=0.3)

# Subplot 2: Ventanas
ax3 = plt.subplot(4, 3, 3)
ax3.plot(t_windows, windows_normal[0], 'b-', linewidth=1.5)
ax3.fill_between(t_windows, 0, windows_normal[0], alpha=0.3, color='blue')
ax3.set_title('Patrón Normal (2s ventana)')
ax3.set_ylim(0, 20)
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(4, 3, 6)
ax4.plot(t_windows, windows_brady[0], 'r-', linewidth=1.5)
ax4.fill_between(t_windows, 0, windows_brady[0], alpha=0.3, color='red')
ax4.set_title('Patrón Bradicinesia (2s ventana)')
ax4.set_ylim(0, 20)
ax4.grid(True, alpha=0.3)

# Subplot 3: Espectro
ax5 = plt.subplot(4, 3, 4)
ax5.semilogy(f_normal, Pxx_normal, 'b-', linewidth=2, label='Normal')
ax5.semilogy(f_brady, Pxx_brady, 'r-', linewidth=2, label='Bradicinesia')
ax5.set_xlim(0, 10)
ax5.set_xlabel('Frecuencia (Hz)')
ax5.set_ylabel('PSD')
ax5.set_title('Espectro de Potencia')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 4: Bandas de frecuencia
ax6 = plt.subplot(4, 3, 5)
bands_power = {'0.5-3Hz': [np.sum(Pxx_normal[(f_normal>=0.5)&(f_normal<=3)]),
                            np.sum(Pxx_brady[(f_brady>=0.5)&(f_brady<=3)])],
               '3-8Hz': [np.sum(Pxx_normal[(f_normal>=3)&(f_normal<=8)]),
                         np.sum(Pxx_brady[(f_brady>=3)&(f_brady<=8)])],
               '8-12Hz': [np.sum(Pxx_normal[(f_normal>=8)&(f_normal<=12)]),
                          np.sum(Pxx_brady[(f_brady>=8)&(f_brady<=12)])]}
x = np.arange(len(bands_power))
width = 0.35
ax6.bar(x - width/2, [b[0] for b in bands_power.values()], width, label='Normal', color='blue')
ax6.bar(x + width/2, [b[1] for b in bands_power.values()], width, label='Bradicinesia', color='red')
ax6.set_xticks(x)
ax6.set_xticklabels(bands_power.keys())
ax6.set_ylabel('Potencia')
ax6.set_title('Potencia por Banda')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Subplot 5: Intervalos
ax7 = plt.subplot(4, 3, 7)
ax7.hist(intervals_normal, bins=15, alpha=0.7, color='blue', label='Normal', density=True)
ax7.hist(intervals_brady, bins=15, alpha=0.7, color='red', label='Bradicinesia', density=True)
ax7.set_xlabel('Intervalo (s)')
ax7.set_ylabel('Densidad')
ax7.set_title('Distribución de Intervalos')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Subplot 6: Espectrograma
ax8 = plt.subplot(4, 3, 8)
f, t_spec, Sxx = spectrogram(acc_brady, fs=FS, nperseg=128, noverlap=64)
im = ax8.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
ax8.set_ylim(0, 12)
ax8.set_xlabel('Tiempo (s)')
ax8.set_ylabel('Frecuencia (Hz)')
ax8.set_title('Espectrograma - Bradicinesia')
plt.colorbar(im, ax=ax8, label='dB')

# Subplot 7: Feature importance
ax9 = plt.subplot(4, 3, 9)
top_features_short = [feature_names[i].replace('_', '\n')[:20] for i in top_indices[:5]]
top_imps = diff[top_indices[:5]]
ax9.barh(top_features_short, top_imps, color='green')
ax9.set_xlabel('Diferencia de medias')
ax9.set_title('Top Features Discriminantes')
ax9.grid(True, alpha=0.3)

# Subplot 8: Resumen clínico
ax10 = plt.subplot(4, 3, 10)
ax10.axis('off')
clinical_text = f"""
RESUMEN CLÍNICO:

Normal:
• Frecuencia tapping: {len(peaks_normal)/30:.1f} Hz
• Intervalo medio: {np.mean(intervals_normal):.2f} s
• Variabilidad: {np.std(intervals_normal):.3f} s
• Potencia temblor: {bands_power['3-8Hz'][0]:.1f}

Bradicinesia:
• Frecuencia tapping: {len(peaks_brady)/30:.1f} Hz
• Intervalo medio: {np.mean(intervals_brady):.2f} s
• Variabilidad: {np.std(intervals_brady):.3f} s
• Potencia temblor: {bands_power['3-8Hz'][1]:.1f}

CARACTERÍSTICAS CLAVE:
• ↓ Frecuencia de movimiento
• ↑ Intervalos entre golpes
• ↑ Variabilidad temporal
• ↓ Potencia en alta frecuencia
"""
ax10.text(0.05, 0.95, clinical_text, transform=ax10.transAxes, fontsize=9, 
          verticalalignment='top', family='monospace')

# Subplot 9: Interpretación biomecánica
ax11 = plt.subplot(4, 3, 11)
ax11.axis('off')
biomech_text = """
INTERPRETACIÓN BIOMECÁNICA:

Bradicinesia = Movimiento Lento

Mecanismo:
• Disminución de la frecuencia de activación muscular
• Incapacidad para generar movimientos rápidos
• Fatiga muscular durante actividad repetitiva

Consecuencias:
• Menor aceleración pico
• Menor velocidad angular
• Mayor irregularidad en el ritmo
• Pérdida de potencia en altas frecuencias

Biomarcadores:
• Jerk (cambio de aceleración) ↓
• Potencia giroscopio en 0.5-3Hz ↓
• Variabilidad intervalo ↑
"""
ax11.text(0.05, 0.95, biomech_text, transform=ax11.transAxes, fontsize=9,
          verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('analysis/paper_summary_figure.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura resumen guardada: analysis/paper_summary_figure.png")

print("\n" + "="*60)
print("✅ ANÁLISIS VISUAL COMPLETADO")
print("="*60)
print("""
Figuras generadas:
  1. time_series_with_symptom_markers.png - Serie temporal con marcadores
  2. symptom_pattern_windows.png - Patrones en ventanas de 2s
  3. frequency_domain_analysis.png - Análisis espectral
  4. top_features_boxplots.png - Features más importantes
  5. spectrogram_analysis.png - Espectrogramas
  6. rhythm_analysis.png - Análisis de ritmo
  7. paper_summary_figure.png - Figura compuesta para paper
""")