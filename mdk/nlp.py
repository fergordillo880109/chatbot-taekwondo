"""Clasificador de intenciones: preprocesamiento + TF-IDF + LinearSVC.

El modelo es pequeño (≈220 frases), así que se entrena al iniciar la API en
menos de un segundo. Esto evita guardar archivos .pkl que dependen de la
versión exacta de scikit-learn y hace las métricas reproducibles: mismos datos
+ mismo código = mismo modelo.
"""
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .datos_intenciones import EJEMPLOS, REFUERZO, AMPLIACION, PRUEBA_INDEPENDIENTE

INTENCIONES = list(EJEMPLOS)
MARGEN_MINIMO = 0.25   # diferencia entre la 1.ª y 2.ª intención para aceptar la predicción

STOPWORDS = set("a al de del el en es la las lo los mi para por que se su sus un una y me te le hay "
                "tienen tiene puedo con como cual cuando donde cuanto desde si".split())


def tokens_de_texto(texto):
    texto = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                    if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-zñ0-9]+", texto)


def stem_ligero(palabra):
    """Reducción de sufijos propia y documentada (no es el SnowballStemmer de NLTK)."""
    for sufijo in ("aciones", "acion", "mente", "ados", "adas", "ando", "iendo", "es", "s"):
        if len(palabra) > len(sufijo) + 4 and palabra.endswith(sufijo):
            return palabra[:-len(sufijo)]
    return palabra


def preprocesar(texto):
    """minúsculas → sin tildes → tokens → sin stopwords → stemming ligero."""
    return [stem_ligero(t) for t in tokens_de_texto(texto) if t not in STOPWORDS]


def _aplanar(diccionario):
    frases = [s for grupo in diccionario.values() for s in grupo]
    etiquetas = [k for k, grupo in diccionario.items() for _ in grupo]
    return frases, etiquetas


def particiones(ampliado=True):
    """Devuelve (x_entrena, y_entrena, x_prueba_original, y_prueba_original).

    La prueba original (18 frases) se obtiene con la MISMA división del avance 2
    para poder comparar. Con ``ampliado=False`` se reproduce el modelo del avance 2.
    """
    frases, etiquetas = _aplanar(EJEMPLOS)
    x_ent, x_pru, y_ent, y_pru = train_test_split(
        frases, etiquetas, test_size=.25, random_state=42, stratify=etiquetas)
    extras = [REFUERZO, AMPLIACION] if ampliado else [REFUERZO]
    for extra in extras:
        f, e = _aplanar(extra)
        x_ent, y_ent = x_ent + f, y_ent + e
    return x_ent, y_ent, x_pru, y_pru


def prueba_independiente():
    return _aplanar(PRUEBA_INDEPENDIENTE)


def crear_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=preprocesar, token_pattern=None,
                                  lowercase=False, ngram_range=(1, 2))),
        ('svm', LinearSVC(random_state=42)),
    ])


class ClasificadorIntenciones:
    def __init__(self, ampliado=True):
        x, y, _, _ = particiones(ampliado)
        self.modelo = crear_pipeline().fit(x, y)
        self.n_entrenamiento = len(x)
        self.clases = list(self.modelo.named_steps['svm'].classes_)

    def predecir(self, texto):
        """Devuelve intención, margen y decisión (aceptada / ambigua / sin_vocabulario)."""
        vector = self.modelo.named_steps['tfidf'].transform([texto])
        if vector.nnz == 0:
            return {'intencion': None, 'margen': 0.0, 'decision': 'sin_vocabulario', 'puntajes': {}}
        puntajes = self.modelo.decision_function([texto])[0]
        orden = np.argsort(puntajes)[::-1]
        margen = float(puntajes[orden[0]] - puntajes[orden[1]])
        intencion = self.clases[orden[0]]
        return {
            'intencion': intencion,
            'margen': round(margen, 3),
            'decision': 'aceptada' if margen >= MARGEN_MINIMO else 'ambigua',
            'puntajes': {c: round(float(p), 3) for c, p in zip(self.clases, puntajes)},
        }
