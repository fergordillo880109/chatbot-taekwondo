"""Pruebas de las decisiones del agente (R0–R9) con un modelo visual simulado."""
import pytest

from evaluacion import casos_agente
from evaluacion.simulacion import imagen_png
from mdk.agente import sanitizar_contexto
from mdk.vision import ImagenInvalida


def test_matriz_completa_de_casos(agente):
    filas = casos_agente.ejecutar(agente, imagen_png)
    fallidas = [(f['caso'], f['turno'], f['esperada'], f['obtenida']) for f in filas if not f['correcta']]
    assert not fallidas


def test_r4_guarda_pregunta_y_r5_la_retoma(agente):
    r1 = agente.atender('Tengo este, ¿lo puedo llevar?')
    assert r1['decision'] == 'R4'
    assert r1['contexto']['pregunta_pendiente'] == 'Tengo este, ¿lo puedo llevar?'
    r2 = agente.atender('', imagen_png('casco_claro'), r1['contexto'])
    assert r2['decision'] == 'R5'
    assert 'retoma_pregunta_pendiente' in r2['plan']
    assert 'pregunta_pendiente' not in r2['contexto']
    assert r2['contexto']['ultimo_articulo'] == 'careta'


def test_r6_no_cambia_memoria_y_ofrece_opciones(agente):
    r = agente.atender('¿Qué es esto?', imagen_png('dudosa'), {'ultimo_articulo': 'uniforme'})
    assert r['decision'] == 'R6'
    assert r['contexto']['ultimo_articulo'] == 'uniforme'
    assert 'otra foto' in r['respuesta']


def test_r9_aviso_de_limites(agente):
    r = agente.atender('¿El peto que tengo es original?')
    assert 'R9' in r['reglas_aplicadas']
    r = agente.atender('¿Es seguro el taekwondo para niños?')        # no habla de un artículo
    assert 'R9' not in r['reglas_aplicadas']


def test_r0_sin_modelo_visual(agente_sin_vision):
    r = agente_sin_vision.atender('¿Cuánto cuesta?', imagen_png('peto_claro'))
    assert r['decision'] == 'R0'
    assert 'no pude analizar la fotografía' in r['respuesta']


def test_imagen_invalida_lanza_error(agente):
    with pytest.raises(ImagenInvalida):
        agente.atender('¿Qué es?', b'esto no es una imagen')


def test_entrada_vacia(agente):
    assert 'Escribe una pregunta' in agente.atender('', None)['respuesta']


def test_contexto_del_cliente_se_sanitiza():
    sucio = {'ultimo_articulo': 'bomba', 'mes_consulta': 99, 'tipo_alumno': 'admin',
             'cerrado': 'si', 'pregunta_pendiente': 'x' * 5000, 'otra_clave': 1}
    limpio = sanitizar_contexto(sucio)
    assert 'ultimo_articulo' not in limpio and 'mes_consulta' not in limpio
    assert 'tipo_alumno' not in limpio and 'otra_clave' not in limpio
    assert limpio['cerrado'] is False and len(limpio['pregunta_pendiente']) == 300
    assert sanitizar_contexto('no es dict') == {}


def test_este_mes_no_es_referencia_a_objeto(agente):
    assert agente.atender('¿Cuánto cuesta este mes?')['decision'] == 'R1'


def test_salida_tiene_formato_estable(agente):
    r = agente.atender('hola')
    for clave in ('respuesta', 'decision', 'tipo_decision', 'ruta', 'confianza', 'explicacion',
                  'siguiente_paso', 'contexto', 'percepcion', 'plan', 'reglas_aplicadas'):
        assert clave in r
