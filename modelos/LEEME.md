# Modelos

- `mdk_clasificador_visual.keras` — EfficientNetB0 con transferencia de aprendizaje (avance 2).
  Guardado el 20/11/2026 con **Keras 3.13.2**. Entrada 224×224×3 (RGB, valores 0–255; el modelo
  incluye su propio reescalado). Salida softmax en el orden `casco, guantes, peto, uniforme`.
  Exactitud en el conjunto de prueba del avance 2: 24/24.
- El clasificador de intenciones (TF-IDF + LinearSVC) **no se guarda**: se entrena al iniciar la API
  a partir de `mdk/datos_intenciones.py` (≈200 frases, menos de un segundo). Así el resultado es
  reproducible y no depende de la versión de scikit-learn con la que se serializó.
