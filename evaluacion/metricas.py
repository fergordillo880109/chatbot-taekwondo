"""Métricas reproducibles para el reporte final.

- evaluar_intenciones():   SVM del avance 2 vs. SVM ampliado, en la prueba original y la independiente.
- evaluar_chatbot_completo(): cómo responde el sistema completo (agente) a la prueba independiente.
- evaluar_vision():        accuracy, precisión, recall, F1 y matriz de confusión de la CNN real.
- prueba_carga():          latencia p50/p95 y errores con 1, 5 y 20 usuarios simultáneos.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mdk import nlp
from mdk.vision import CLASES

RESULTADOS = Path('resultados')


def _guardar_matriz(y_real, y_pred, etiquetas, titulo, archivo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    RESULTADOS.mkdir(exist_ok=True)
    fig, eje = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_real, y_pred, labels=etiquetas, cmap='Blues',
                                            xticks_rotation=45, ax=eje, colorbar=False)
    eje.set_title(titulo)
    fig.tight_layout()
    fig.savefig(RESULTADOS / archivo, dpi=150)
    plt.close(fig)
    return RESULTADOS / archivo


# ----------------------------------------------------------------------------- intenciones
def evaluar_intenciones(guardar=True):
    filas, detalle = [], {}
    xi, yi = nlp.prueba_independiente()
    for nombre, ampliado in (('Avance 2 (82 frases)', False), ('Final ampliado', True)):
        x, y, xp, yp = nlp.particiones(ampliado)
        modelo = nlp.crear_pipeline().fit(x, y)
        for conjunto, xs, ys in (('Prueba original (18)', xp, yp), ('Prueba independiente (60)', xi, yi)):
            pred = modelo.predict(xs)
            rep = classification_report(ys, pred, output_dict=True, zero_division=0)
            filas.append({'modelo': nombre, 'frases_entrenamiento': len(x), 'conjunto': conjunto,
                          'aciertos': int(sum(a == b for a, b in zip(ys, pred))), 'total': len(ys),
                          'accuracy': round(accuracy_score(ys, pred), 3),
                          'precision_macro': round(rep['macro avg']['precision'], 3),
                          'recall_macro': round(rep['macro avg']['recall'], 3),
                          'f1_macro': round(rep['macro avg']['f1-score'], 3)})
            detalle[(nombre, conjunto)] = (ys, pred, xs)
    resumen = pd.DataFrame(filas)
    ys, pred, xs = detalle[('Final ampliado', 'Prueba independiente (60)')]
    por_clase = pd.DataFrame(classification_report(ys, pred, output_dict=True, zero_division=0)).T.round(3)
    errores = pd.DataFrame([{'frase': f, 'real': a, 'predicha': b} for f, a, b in zip(xs, ys, pred) if a != b])
    if guardar:
        RESULTADOS.mkdir(exist_ok=True)
        resumen.to_csv(RESULTADOS / 'intenciones_resumen.csv', index=False)
        por_clase.to_csv(RESULTADOS / 'intenciones_por_clase.csv')
        errores.to_csv(RESULTADOS / 'intenciones_errores.csv', index=False)
        _guardar_matriz(ys, pred, nlp.INTENCIONES, 'Intenciones · prueba independiente',
                        'matriz_intenciones.png')
    return resumen, por_clase, errores


# ----------------------------------------------------------------------------- sistema completo
# Temas de las reglas que se consideran coherentes con cada etiqueta de la prueba.
COHERENTES = {
    'horarios': {'horarios', 'fin_semana', 'edades'},
    'costos': {'costos', 'mensualidad', 'evaluaciones', 'promocion', 'equipamiento', 'inscripcion'},
    'inscripcion': {'inscripcion', 'documentos', 'experiencia', 'clase_muestra', 'visita'},
    'ubicacion': {'ubicacion'},
    'uniforme': {'uniforme', 'equipamiento'},
    'proteccion': {'proteccion', 'equipamiento'},
}


def evaluar_chatbot_completo(agente, guardar=True):
    """Pasa cada frase de la prueba independiente por el AGENTE (reglas → SVM → humano).

    Criterio: la respuesta es «coherente» si las intenciones de reglas usadas o la
    intención del SVM corresponden al tema de la etiqueta (tabla COHERENTES).
    Es una medida automática aproximada; conviene revisar manualmente una muestra.
    """
    filas = []
    for etiqueta, frases in nlp.PRUEBA_INDEPENDIENTE.items():
        for frase in frases:
            t0 = time.perf_counter()
            r = agente.atender(frase)
            latencia = time.perf_counter() - t0
            temas = set(r['percepcion']['intenciones_reglas'])
            if r['decision'] == 'R2':
                temas = {r['nlp']['intencion']}
            filas.append({'frase': frase, 'etiqueta': etiqueta, 'decision': r['decision'],
                          'temas': ', '.join(sorted(temas)),
                          'derivada_a_humano': r['decision'] == 'R3',
                          'coherente': bool(temas & COHERENTES[etiqueta]),
                          'latencia_ms': round(latencia * 1000, 2), 'respuesta': r['respuesta']})
    tabla = pd.DataFrame(filas)
    resumen = {
        'frases': len(tabla),
        'respondidas_por_reglas_R1': int((tabla.decision == 'R1').sum()),
        'respondidas_por_svm_R2': int((tabla.decision == 'R2').sum()),
        'derivadas_a_humano_R3': int((tabla.decision == 'R3').sum()),
        'respuestas_coherentes': int(tabla.coherente.sum()),
        'tasa_coherencia': round(float(tabla.coherente.mean()), 3),
        'latencia_ms_p50': round(float(tabla.latencia_ms.median()), 2),
        'latencia_ms_p95': round(float(tabla.latencia_ms.quantile(.95)), 2),
    }
    if guardar:
        RESULTADOS.mkdir(exist_ok=True)
        tabla.to_csv(RESULTADOS / 'chatbot_sistema_completo.csv', index=False)
        pd.Series(resumen).to_csv(RESULTADOS / 'chatbot_resumen.csv', header=['valor'])
    return resumen, tabla


# ----------------------------------------------------------------------------- visión
def evaluar_vision(clasificador, carpeta, nombre='prueba', guardar=True):
    """``carpeta`` contiene subcarpetas casco/, guantes/, peto/, uniforme/ con fotos."""
    carpeta = Path(carpeta)
    filas = []
    for clase in CLASES:
        for ruta in sorted((carpeta / clase).glob('*')):
            if ruta.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            r = clasificador.clasificar(ruta.read_bytes())
            filas.append({'archivo': ruta.name, 'real': clase, 'predicha': r['clase'],
                          'confianza': r['confianza'], 'latencia_ms': r['latencia_modelo_s'] * 1000,
                          'correcta': clase == r['clase']})
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        raise ValueError(f'No se encontraron imágenes en {carpeta}/<clase>/')
    rep = classification_report(tabla.real, tabla.predicha, labels=CLASES, output_dict=True, zero_division=0)
    resumen = {'conjunto': nombre, 'imagenes': len(tabla), 'aciertos': int(tabla.correcta.sum()),
               'accuracy': round(float(tabla.correcta.mean()), 3),
               'precision_macro': round(rep['macro avg']['precision'], 3),
               'recall_macro': round(rep['macro avg']['recall'], 3),
               'f1_macro': round(rep['macro avg']['f1-score'], 3),
               'confianza_promedio': round(float(tabla.confianza.mean()), 3),
               'imagenes_confianza_menor_0.65': int((tabla.confianza < 0.65).sum()),
               'latencia_ms_p50': round(float(tabla.latencia_ms.median()), 1)}
    por_clase = pd.DataFrame(rep).T.round(3)
    if guardar:
        RESULTADOS.mkdir(exist_ok=True)
        tabla.to_csv(RESULTADOS / f'vision_{nombre}_detalle.csv', index=False)
        por_clase.to_csv(RESULTADOS / f'vision_{nombre}_por_clase.csv')
        _guardar_matriz(tabla.real, tabla.predicha, CLASES, f'Clasificador visual · {nombre}',
                        f'matriz_vision_{nombre}.png')
    return resumen, por_clase, tabla


# ----------------------------------------------------------------------------- carga
def prueba_carga(url, imagen_bytes, usuarios=(1, 5, 20), solicitudes_por_usuario=3, guardar=True):
    import requests

    def una(_):
        t0 = time.perf_counter()
        try:
            r = requests.post(f'{url}/agente', data={'mensaje': '¿Cuánto cuesta?', 'contexto': '{}'},
                              files={'file': ('foto.jpg', imagen_bytes, 'image/jpeg')}, timeout=120)
            codigo = r.status_code
        except requests.exceptions.RequestException:
            codigo = 0
        return time.perf_counter() - t0, codigo

    filas, crudas = [], {}
    for n in usuarios:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(una, range(n * solicitudes_por_usuario)))
        total = time.perf_counter() - t0
        lat = np.array([r[0] for r in res]) * 1000
        crudas[n] = lat
        filas.append({'usuarios_simultaneos': n, 'solicitudes': len(res),
                      'p50_ms': round(float(np.percentile(lat, 50))), 'p95_ms': round(float(np.percentile(lat, 95))),
                      'max_ms': round(float(lat.max())), 'errores': sum(r[1] != 200 for r in res),
                      'solicitudes_por_segundo': round(len(res) / total, 2)})
    tabla = pd.DataFrame(filas)
    if guardar:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        RESULTADOS.mkdir(exist_ok=True)
        tabla.to_csv(RESULTADOS / 'prueba_carga.csv', index=False)
        fig, eje = plt.subplots(figsize=(6, 3.5))
        eje.boxplot(list(crudas.values()))
        eje.set_xticks(range(1, len(crudas) + 1), [f'{u} usuarios' for u in crudas])
        eje.set_ylabel('Latencia de extremo a extremo (ms)')
        eje.set_title('Prueba de carga · /agente con imagen')
        eje.grid(axis='y', alpha=.3)
        fig.tight_layout()
        fig.savefig(RESULTADOS / 'prueba_carga.png', dpi=150)
        plt.close(fig)
    return tabla


def fecha_fija():
    """Fecha usada en las pruebas para que las promociones den siempre el mismo resultado."""
    return date(2026, 11, 27)
