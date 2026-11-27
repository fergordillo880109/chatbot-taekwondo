"""Clasificador visual: EfficientNetB0 (transferencia de aprendizaje, avance 2).

- El modelo .keras se carga UNA vez (en la API) y se reutiliza.
- El preprocesamiento replica el del entrenamiento: RGB → 224×224 (bilineal,
  como ``image_dataset_from_directory``) → float32 en rango 0–255. La
  EfficientNetB0 de Keras ya incluye su propio reescalado y normalización.
- TensorFlow se importa solo al cargar el modelo; así el resto del sistema
  (chatbot, agente, pruebas) funciona aunque la visión no esté disponible.
"""
import io
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

CLASES = ['casco', 'guantes', 'peto', 'uniforme']           # mismo orden del entrenamiento
TAMANO = (224, 224)
FORMATOS_PERMITIDOS = {'JPEG', 'PNG'}
RUTA_MODELO = Path(__file__).resolve().parent.parent / 'modelos' / 'mdk_clasificador_visual.keras'

Image.MAX_IMAGE_PIXELS = 40_000_000    # evita «bombas de descompresión»


class ImagenInvalida(ValueError):
    """La entrada no es una imagen JPG/PNG legible."""


def leer_imagen(image_bytes):
    """Valida que los bytes sean realmente una imagen JPG o PNG y la devuelve en RGB."""
    if not image_bytes:
        raise ImagenInvalida('El archivo está vacío.')
    try:
        with Image.open(io.BytesIO(image_bytes)) as prueba:
            formato = prueba.format
            prueba.verify()                      # detecta archivos corruptos o falsos
        if formato not in FORMATOS_PERMITIDOS:
            raise ImagenInvalida('Formato no permitido. Usa JPG o PNG.')
        imagen = Image.open(io.BytesIO(image_bytes))
        imagen.load()
        return imagen.convert('RGB')
    except ImagenInvalida:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImagenInvalida('El archivo no es una imagen válida.') from exc


def preprocesar(imagen):
    arreglo = np.asarray(imagen.resize(TAMANO, Image.BILINEAR), dtype=np.float32)
    return arreglo[None, ...]                    # (1, 224, 224, 3)


class ClasificadorVisual:
    def __init__(self, ruta=RUTA_MODELO, modelo=None):
        """``modelo`` permite inyectar un modelo falso en las pruebas automáticas."""
        self.ruta = Path(ruta)
        self.modelo = modelo
        self.error = None
        self._candado = threading.Lock()         # una inferencia a la vez por proceso

    def cargar(self):
        if self.modelo is not None:
            return True
        try:
            import keras                         # Keras 3 (backend TensorFlow)
            self.modelo = keras.models.load_model(self.ruta, compile=False)
            self.modelo.predict(np.zeros((1, *TAMANO, 3), dtype=np.float32), verbose=0)  # calentamiento
            return True
        except Exception as exc:                 # se informa en /health, no se detiene la API
            self.error = f'{type(exc).__name__}: {exc}'
            self.modelo = None
            return False

    @property
    def disponible(self):
        return self.modelo is not None

    def clasificar(self, image_bytes):
        if not self.disponible:
            raise RuntimeError('El clasificador visual no está disponible.')
        imagen = leer_imagen(image_bytes)
        lote = preprocesar(imagen)
        t0 = time.perf_counter()
        with self._candado:
            salida = self.modelo.predict(lote, verbose=0)
        latencia = time.perf_counter() - t0
        probabilidades = np.asarray(salida, dtype=float).reshape(-1)
        idx = int(np.argmax(probabilidades))
        return {
            'clase': CLASES[idx],
            'confianza': round(float(probabilidades[idx]), 4),
            'probabilidades': {c: round(float(p), 4) for c, p in zip(CLASES, probabilidades)},
            'latencia_modelo_s': round(latencia, 4),
            'tamano_original': list(imagen.size),
        }
