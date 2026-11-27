"""Pruebas de la API Flask (validación, códigos HTTP, contexto y métricas)."""
import io
import json

import pytest

from evaluacion.simulacion import ModeloFalso, imagen_png


@pytest.fixture(scope='module')
def cliente(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.chdir(tmp_path_factory.mktemp('api'))      # logs/ en carpeta temporal
    import api
    api.VISION.modelo = ModeloFalso()                              # modelo simulado (sin TensorFlow)
    api.app.config['TESTING'] = True
    return api.app.test_client()


@pytest.fixture(scope='module')
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def archivo(nombre, datos):
    return {'file': (io.BytesIO(datos), nombre)}


def test_health_e_inicio(cliente):
    assert cliente.get('/').status_code == 200
    salud = cliente.get('/health').get_json()
    assert salud['estado'] == 'ok' and salud['componentes']['clasificador_visual']


def test_chat_ok_y_vacio(cliente):
    r = cliente.post('/chat', json={'mensaje': '¿Qué días hay clases?'})
    assert r.status_code == 200 and r.get_json()['decision'] == 'R1'
    assert cliente.post('/chat', json={}).status_code == 400
    assert cliente.post('/chat', json={'mensaje': 'x' * 600}).status_code == 400


def test_predict_image(cliente):
    r = cliente.post('/predict-image', data=archivo('peto.png', imagen_png('peto_claro')),
                     content_type='multipart/form-data')
    datos = r.get_json()
    assert r.status_code == 200 and datos['clase'] == 'peto' and datos['nivel_confianza'] == 'alta'


@pytest.mark.parametrize('nombre,datos,codigo', [
    ('notas.txt', b'hola', 400),                        # extensión no permitida
    ('falsa.png', b'esto no es una imagen', 400),       # dice PNG pero no lo es
    ('vacia.jpg', b'', 400),                            # archivo vacío
    ('enorme.png', b'0' * (6 * 1024 * 1024), 413),      # > 5 MB
])
def test_predict_image_errores(cliente, nombre, datos, codigo):
    r = cliente.post('/predict-image', data=archivo(nombre, datos), content_type='multipart/form-data')
    assert r.status_code == codigo and 'error' in r.get_json()


def test_predict_image_sin_archivo(cliente):
    assert cliente.post('/predict-image', data={}, content_type='multipart/form-data').status_code == 400


def test_agente_conversacion_con_contexto(cliente):
    r1 = cliente.post('/agente', data={'mensaje': 'Tengo este, ¿lo puedo llevar?', 'contexto': '{}'},
                      content_type='multipart/form-data').get_json()
    assert r1['decision'] == 'R4'
    datos = {'mensaje': '', 'contexto': json.dumps(r1['contexto'])}
    datos.update(archivo('casco.png', imagen_png('casco_claro')))
    r2 = cliente.post('/agente', data=datos, content_type='multipart/form-data').get_json()
    assert r2['decision'] == 'R5' and r2['vision']['clase'] == 'casco'
    r3 = cliente.post('/agente', json={'mensaje': '¿cuánto cuesta?', 'contexto': r2['contexto']}).get_json()
    assert r3['decision'] == 'R8' and '$850' in r3['respuesta']


def test_agente_entradas_invalidas(cliente):
    assert cliente.post('/agente', json={'mensaje': ''}).status_code == 400
    r = cliente.post('/agente', data={'mensaje': 'hola', 'contexto': '{no es json'},
                     content_type='multipart/form-data')
    assert r.status_code == 200                                     # contexto corrupto se ignora
    datos = {'mensaje': '¿qué es?'}
    datos.update(archivo('x.png', b'no imagen'))
    assert cliente.post('/agente', data=datos, content_type='multipart/form-data').status_code == 400


def test_vision_no_disponible(cliente):
    import api
    modelo = api.VISION.modelo
    api.VISION.modelo = None
    try:
        r = cliente.post('/predict-image', data=archivo('p.png', imagen_png('peto_claro')),
                         content_type='multipart/form-data')
        assert r.status_code == 503
        assert cliente.get('/health').get_json()['estado'] == 'degradado'
    finally:
        api.VISION.modelo = modelo


def test_metricas(cliente):
    m = cliente.get('/metrics').get_json()
    assert m['solicitudes_totales'] > 0 and 'R4' in m['decisiones_agente']
    assert m['codigos_http'].get('413', 0) >= 1 and m['latencia_ms']['p95'] >= 0
