# BIOCLITE-PUCP Dataset Description

## 1. Overview

Dataset basado en BIOCLITE para el estudio de síntomas motores en pacientes con Parkinson usando smartwatch.

- Fuente original: BIOCLITE (Zenodo)
- Frecuencia de muestreo: 50 Hz
- Sensores:
  - Acelerómetro triaxial (m/s²)
  - Giroscopio triaxial (rad/s)

## 2. Participantes

- 24 pacientes con Parkinson
- 16 sujetos control
- Total: 40 participantes

## 3. Protocolo Experimental

Cada participante realizó:

- 8 ejercicios motores
- Contextos:
  - Supervisado (inicio y final)
  - No supervisado (7 días consecutivos)

## 4. Estructura de los Datos

### Raw Data

Formato original: `.mat`

Contiene:
- Señales IMU
- Identificador de sujeto
- Tipo de sesión (supervised / unsupervised)
- Ejercicio

### Estructura interna (post-conversión sugerida)

```bash
subject_id/
    session_type/
        day_X/
            exercise_Y.csv
