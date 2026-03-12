# BIOCLITE-PUCP: Clasificación de Parkinson con Deep Learning y XAI a partir de Señales de Smartwatch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tu-usuario/BIOCLITE-PUCP/blob/main/notebooks/01_EDA_y_Preprocesamiento.ipynb)

## 📝 Descripción General

Este repositorio contiene el código para un pipeline completo de Deep Learning y técnicas de Inteligencia Artificial Explicable (XAI) aplicado a la clasificación de la Enfermedad de Parkinson. El proyecto utiliza el dataset público **BIOCLITE v2**, que contiene señales de acelerómetro y giroscopio de smartwatch de 40 sujetos (16 controles sanos y 24 pacientes con Parkinson) mientras realizan 8 ejercicios estandarizados del MDS-UPDRS.

**Objetivo Principal:** Desarrollar y validar rigurosamente un modelo robusto que pueda distinguir entre sujetos con Parkinson y controles sanos, e identificar qué aspectos de la señal (sensores, momentos temporales, ejercicios) son más determinantes para el diagnóstico.

## 🚀 Características Principales

*   **Validación Robusta:** Implementación de **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** para asegurar que el modelo generaliza a sujetos no vistos y evitar el "data leakage", un problema crítico en el procesamiento de señales biomédicas.
*   **Modelos Comparados:** Se evaluaron y compararon dos enfoques:
    1.  **Machine Learning Clásico:** Extracción de características ingenieriles (estadísticas en tiempo y frecuencia) + **XGBoost** (baseline).
    2.  **Deep Learning:** Modelo híbrido **CNN-LSTM** optimizado con Keras Tuner para capturar patrones espacio-temporales en las señales.
*   **Análisis Multidimensional:** Evaluación del rendimiento del modelo no solo de forma global, sino también segmentada por:
    *   **Tipo de Ejercicio:** Identificando cuáles de los 8 ejercicios del MDS-UPDRS son más discriminativos.
    *   **Contexto de la Sesión:** Comparando el rendimiento en entornos supervisados (clínica) vs. no supervisados (vida diaria).
    *   **Severidad de la Enfermedad:** Analizando la correlación de las predicciones con la puntuación MDS-UPDRS.
*   **Inteligencia Artificial Explicable (XAI):** Se aplicaron técnicas de vanguardia para abrir la "caja negra" del modelo:
    *   **SHAP (DeepExplainer):** Para determinar la importancia global de cada sensor (acelerómetro vs. giroscopio).
    *   **Grad-CAM:** Para visualizar en qué instantes de tiempo la red convolutional se enfoca más para tomar una decisión.
    *   **Mecanismo de Atención:** Implementado en una variante del modelo para visualizar los pesos de atención a lo largo de la señal.

## 🧠 Principales Hallazgos

Los resultados obtenidos tras una validación rigurosa (LOSO-CV) son los siguientes:

1.  **Rendimiento del Modelo:**
    *   **Accuracy:** `79.3% ± 23.5%`
    *   **F1-Score:** `59.1% ± 48.3%`
    *   Estos valores reflejan la alta variabilidad entre sujetos. El modelo es excelente identificando Parkinson (precisión media del 97.3%), pero tiene una alta tasa de error en la clasificación de controles sanos, lo que indica un sesgo hacia la clase mayoritaria.

2.  **Ejercicios Más Relevantes (Top 3):**
    Los ejercicios que mejor permiten diferenciar entre ambos grupos son:
    1.  **Ejercicio 6 (Pronación/Supinación - MDS-UPDRS 3.6):** `Accuracy 95.6%`
    2.  **Ejercicio 5 (Abrir/Cerrar Mano - MDS-UPDRS 3.5):** `Accuracy 93.7%`
    3.  **Ejercicio 8 (Marcha - MDS-UPDRS 3.10):** `Accuracy 93.1%`
    Esto sugiere que estos ejercicios son biomarcadores digitales muy potentes.

3.  **Importancia de Sensores y Variables Clínicas:**
    *   **SHAP** reveló que los sensores de **giroscopio** (especialmente Gyro_X y Gyro_Y) son, en promedio, más importantes que el acelerómetro para la clasificación, destacando la relevancia de la velocidad angular en el análisis del movimiento parkinsoniano.
    *   Se observó una **correlación positiva y significativa (`r=0.49, p<0.001`)** entre la probabilidad predicha por el modelo y la puntuación UPDRS, lo que indica que el modelo es sensible a la severidad de los síntomas.

4.  **Validez Ecológica:**
    *   El modelo mantiene un rendimiento similar en contextos **no supervisados (vida real, Acc 66.5%)** y **supervisados (entorno clínico, Acc 66.3%)**. Este hallazgo es crucial, ya que valida el potencial de usar smartwatches para el monitoreo continuo y pasivo de los pacientes en su hogar.

5.  **Advertencia sobre Data Leakage:**
    *   Se demostró cuantitativamente que una validación incorrecta (mezclando ventanas del mismo sujeto en train y test) puede **sobreestimar el rendimiento del modelo en más de un 60%** en algunos ejercicios. Esto subraya la necesidad crítica de la validación cruzada por sujetos en este tipo de estudios.

## 📁 Dataset

*   **Fuente:** `BIOCLITE_v2.csv` (debe ser cargado en Google Drive).
*   **Contenido:** Señales crudas de 6 ejes (Acelerómetro y Giroscopio) muestreadas a 50 Hz, junto con metadatos de sesión, sujeto, ejercicio y puntuación UPDRS.

## 🛠️ Instalación y Uso

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/BIOCLITE-PUCP.git
    cd BIOCLITE-PUCP