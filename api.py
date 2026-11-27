"""API REST (Flask) del asistente MDK Tuchtlán.

Endpoints
---------
GET  /               descripción del servicio
GET  /health         estado de la API y de los modelos
GET  /metrics        solicitudes, errores, latencia p50/p95, confianza, decisiones del agente
POST /agente         PRINCIPAL: texto + imagen opcional + contexto → respuesta coordinada
POST /chat           solo texto (pasa por el agente)
POST /predict-image  solo el clasificador visual

Ejecutar:  python api.py            (desarrollo, puerto 5000)
           gunicorn api:app -w 1 --threads 4 -b 0.0.0.0:5000
"""
import json
import logging
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from mdk.agente import AgenteMDK, UMBRAL_VISUAL
from mdk.nlp import ClasificadorIntenciones
from mdk.vision import ClasificadorVisual, ImagenInvalida

VERSION = '1.0.0'
LIMITE_BYTES = 5 * 1024 * 1024
EXTENSIONES = {'jpg', 'jpeg', 'png'}
MAX_TEXTO = 500

# ---------------------------------------------------------------- registro (logs)
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(filename='logs/api.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('mdk')
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# ---------------------------------------------------------------- modelos: se cargan UNA vez
INICIO = time.time()
NLP = ClasificadorIntenciones()
VISION = ClasificadorVisual(os.getenv('MODELO_VISUAL', ClasificadorVisual().ruta))
VISION.cargar()
AGENTE = AgenteMDK(NLP, VISION)
if VISION.error:
    log.error('modelo_visual_no_cargado error=%s', VISION.error)
log.info('api_iniciada vision=%s svm_frases=%s', VISION.disponible, NLP.n_entrenamiento)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = LIMITE_BYTES + 64 * 1024     # margen para los campos del formulario
app.json.ensure_ascii = False


# ---------------------------------------------------------------- métricas en memoria
class Metricas:
    """En producción esto iría a Prometheus/CloudWatch; aquí basta para el prototipo."""

    def __init__(self):
        self.candado = threading.Lock()
        self.solicitudes = Counter()
        self.codigos = Counter()
        self.decisiones = Counter()
        self.clases = Counter()
        self.latencias = deque(maxlen=2000)
        self.confianzas = deque(maxlen=2000)
        self.confianza_baja = 0

    def registrar(self, endpoint, codigo, latencia, decision=None, vision=None):
        with self.candado:
            self.solicitudes[endpoint] += 1
            self.codigos[str(codigo)] += 1
            self.latencias.append(latencia)
            if decision:
                self.decisiones[decision] += 1
            if vision:
                self.clases[vision['clase']] += 1
                self.confianzas.append(vision['confianza'])
                if vision['confianza'] < UMBRAL_VISUAL:
                    self.confianza_baja += 1

    def resumen(self):
        with self.candado:
            lat = np.array(self.latencias) * 1000 if self.latencias else np.array([0.0])
            total = sum(self.solicitudes.values())
            errores = sum(n for c, n in self.codigos.items() if not c.startswith('2'))
            return {
                'tiempo_activo_min': round((time.time() - INICIO) / 60, 1),
                'solicitudes_totales': total,
                'solicitudes_por_endpoint': dict(self.solicitudes),
                'codigos_http': dict(self.codigos),
                'tasa_error': round(errores / total, 3) if total else 0.0,
                'latencia_ms': {'p50': round(float(np.percentile(lat, 50)), 1),
                                'p95': round(float(np.percentile(lat, 95)), 1),
                                'max': round(float(lat.max()), 1)},
                'decisiones_agente': dict(self.decisiones),
                'clases_predichas': dict(self.clases),
                'confianza_visual_promedio': (round(float(np.mean(self.confianzas)), 3)
                                              if self.confianzas else None),
                'imagenes_con_confianza_baja': self.confianza_baja,
            }


METRICAS = Metricas()


def responder(endpoint, t0, cuerpo, codigo=200, decision=None, vision=None, **datos_log):
    latencia = time.perf_counter() - t0
    METRICAS.registrar(endpoint, codigo, latencia, decision, vision)
    # Se registra QUÉ pasó, nunca la imagen ni el texto completo del usuario (privacidad).
    extra = ' '.join(f'{k}={v}' for k, v in datos_log.items())
    log.info('endpoint=%s codigo=%s latencia_ms=%.1f decision=%s clase=%s confianza=%s %s',
             endpoint, codigo, latencia * 1000, decision,
             vision['clase'] if vision else None, vision['confianza'] if vision else None, extra)
    return jsonify(cuerpo), codigo


def error(endpoint, t0, mensaje, codigo, motivo):
    return responder(endpoint, t0, {'error': mensaje}, codigo, motivo=motivo)


# ---------------------------------------------------------------- validación de entradas
def leer_archivo(obligatorio):
    """Devuelve (bytes | None, mensaje_error | None, motivo)."""
    if 'file' not in request.files:
        return None, ("No se recibió el campo 'file' con la imagen." if obligatorio else None), 'sin_archivo'
    archivo = request.files['file']
    nombre = secure_filename(archivo.filename or '')
    if not nombre or '.' not in nombre or nombre.rsplit('.', 1)[1].lower() not in EXTENSIONES:
        return None, 'Formato no permitido. Usa una imagen JPG, JPEG o PNG.', 'extension'
    datos = archivo.read(LIMITE_BYTES + 1)
    if len(datos) > LIMITE_BYTES:
        return None, 'La imagen supera el límite de 5 MB.', 'tamano'
    if not datos:
        return None, 'El archivo está vacío.', 'vacio'
    return datos, None, 'ok'


def leer_contexto(valor):
    if isinstance(valor, dict):
        return valor
    if isinstance(valor, str) and valor:
        try:
            return json.loads(valor)
        except (json.JSONDecodeError, RecursionError):
            return {}
    return {}


def resumen_agente(r):
    """Respuesta pública del agente (todo serializable a JSON)."""
    return {k: r[k] for k in ('respuesta', 'decision', 'tipo_decision', 'reglas_aplicadas', 'ruta',
                              'confianza', 'explicacion', 'siguiente_paso', 'vision', 'nlp',
                              'contexto', 'plan', 'percepcion')}


# ---------------------------------------------------------------- endpoints
@app.get('/')
def inicio():
    return jsonify({'servicio': 'API del asistente MDK Tuchtlán', 'version': VERSION,
                    'endpoints': ['/health', '/metrics', '/agente', '/chat', '/predict-image']})


@app.get('/health')
def health():
    estado = 'ok' if VISION.disponible else 'degradado'
    return jsonify({'estado': estado, 'version': VERSION,
                    'componentes': {'chatbot_reglas': True, 'clasificador_intenciones': True,
                                    'clasificador_visual': VISION.disponible},
                    'detalle_visual': None if VISION.disponible else 'modelo no cargado',
                    'tiempo_activo_s': round(time.time() - INICIO)})


@app.get('/metrics')
def metrics():
    return jsonify(METRICAS.resumen())


@app.post('/agente')
def agente():
    t0 = time.perf_counter()
    if request.is_json:
        datos = request.get_json(silent=True) or {}
        imagen = None
    else:
        datos = request.form
        imagen, mensaje_error, motivo = leer_archivo(obligatorio=False)
        if mensaje_error:
            return error('/agente', t0, mensaje_error, 400, motivo)
    mensaje = str(datos.get('mensaje', '') or '').strip()
    if len(mensaje) > MAX_TEXTO:
        return error('/agente', t0, f'El mensaje supera {MAX_TEXTO} caracteres.', 400, 'texto_largo')
    if not mensaje and imagen is None:
        return error('/agente', t0, 'Envía un mensaje, una imagen o ambos.', 400, 'entrada_vacia')
    try:
        r = AGENTE.atender(mensaje, imagen, leer_contexto(datos.get('contexto')))
    except ImagenInvalida as exc:
        return error('/agente', t0, f'{exc} Prueba con otra foto JPG o PNG.', 400, 'imagen_invalida')
    return responder('/agente', t0, resumen_agente(r), 200, r['decision'], r['vision'],
                     ruta='>'.join(r['ruta']))


@app.post('/chat')
def chat():
    t0 = time.perf_counter()
    datos = request.get_json(silent=True) or {}
    mensaje = str(datos.get('mensaje', '') or '').strip()
    if not mensaje:
        return error('/chat', t0, "Envía un JSON con el campo 'mensaje' no vacío.", 400, 'entrada_vacia')
    if len(mensaje) > MAX_TEXTO:
        return error('/chat', t0, f'El mensaje supera {MAX_TEXTO} caracteres.', 400, 'texto_largo')
    r = AGENTE.atender(mensaje, None, leer_contexto(datos.get('contexto')))
    return responder('/chat', t0, resumen_agente(r), 200, r['decision'], ruta='>'.join(r['ruta']))


@app.post('/predict-image')
def predict_image():
    t0 = time.perf_counter()
    imagen, mensaje_error, motivo = leer_archivo(obligatorio=True)
    if mensaje_error:
        return error('/predict-image', t0, mensaje_error, 400, motivo)
    if not VISION.disponible:
        return error('/predict-image', t0, 'El clasificador de imágenes no está disponible en este momento.',
                     503, 'vision_no_disponible')
    try:
        v = VISION.clasificar(imagen)
    except ImagenInvalida as exc:
        return error('/predict-image', t0, f'{exc} Prueba con otra foto JPG o PNG.', 400, 'imagen_invalida')
    v['nivel_confianza'] = ('alta' if v['confianza'] >= 0.80 else
                            'media' if v['confianza'] >= UMBRAL_VISUAL else 'baja')
    return responder('/predict-image', t0, v, 200, vision=v)


# ---------------------------------------------------------------- manejo de errores
@app.errorhandler(413)
def demasiado_grande(_):
    METRICAS.registrar(request.path, 413, 0.0)
    log.warning('endpoint=%s codigo=413 motivo=tamano', request.path)
    return jsonify({'error': 'La imagen supera el límite de 5 MB.'}), 413


@app.errorhandler(HTTPException)
def http_error(exc):
    METRICAS.registrar(request.path, exc.code, 0.0)
    return jsonify({'error': exc.description}), exc.code


@app.errorhandler(Exception)
def error_interno(exc):
    METRICAS.registrar(request.path, 500, 0.0)
    log.exception('endpoint=%s codigo=500 error_no_controlado', request.path)   # detalle solo en el log
    return jsonify({'error': 'No pudimos procesar la solicitud. Intenta de nuevo en unos minutos.'}), 500


if __name__ == '__main__':
    app.run(host=os.getenv('API_HOST', '127.0.0.1'), port=int(os.getenv('API_PORT', 5000)), threaded=True)
