"""Modelo visual SIMULADO para probar el agente y la API sin TensorFlow.

Genera imágenes PNG de un color y un modelo falso que devuelve probabilidades
según ese color. Sirve para pruebas automáticas (pytest) de la lógica de
decisión. Las métricas del clasificador REAL se calculan en el notebook con el
modelo .keras y fotografías reales.
"""
import io

import numpy as np
from PIL import Image

COLORES = {   # nombre → (color RGB, probabilidades [casco, guantes, peto, uniforme])
    'casco_claro':    ((200, 30, 30),  [0.93, 0.03, 0.03, 0.01]),
    'peto_claro':     ((30, 30, 200),  [0.02, 0.03, 0.92, 0.03]),
    'guantes_claro':  ((200, 200, 30), [0.04, 0.90, 0.04, 0.02]),
    'uniforme_claro': ((240, 240, 240), [0.01, 0.02, 0.02, 0.95]),
    'dudosa':         ((90, 90, 90),   [0.40, 0.05, 0.45, 0.10]),
}


def imagen_png(nombre, tamano=(320, 240)):
    color = COLORES[nombre][0]
    buffer = io.BytesIO()
    Image.new('RGB', tamano, color).save(buffer, format='PNG')
    return buffer.getvalue()


class ModeloFalso:
    def predict(self, lote, verbose=0):
        color = tuple(int(round(v)) for v in np.asarray(lote)[0].mean(axis=(0, 1)))
        for rgb, probs in COLORES.values():
            if max(abs(a - b) for a, b in zip(rgb, color)) <= 3:
                return np.array([probs], dtype=np.float32)
        return np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
