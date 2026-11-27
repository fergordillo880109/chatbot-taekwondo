"""Pruebas del clasificador de intenciones y de la reproducibilidad de sus métricas."""
from sklearn.metrics import accuracy_score

from mdk import nlp


def test_reproduce_resultado_del_avance_2():
    x, y, xp, yp = nlp.particiones(ampliado=False)
    modelo = nlp.crear_pipeline().fit(x, y)
    assert len(x) == 82 and len(xp) == 18
    assert round(accuracy_score(yp, modelo.predict(xp)), 3) == 0.889     # 16/18 del avance 2


def test_conjunto_ampliado_tiene_al_menos_35_frases_por_intencion():
    for intencion in nlp.INTENCIONES:
        total = sum(len(d[intencion]) for d in (nlp.EJEMPLOS, nlp.REFUERZO, nlp.AMPLIACION))
        assert total >= 35, intencion


def test_prueba_independiente_no_se_usa_para_entrenar():
    x, _, _, _ = nlp.particiones()
    xi, _ = nlp.prueba_independiente()
    normal = lambda s: ''.join(c for c in s.lower() if c.isalnum())
    assert not {normal(s) for s in x} & {normal(s) for s in xi}


def test_prediccion_y_margen(nlp):
    r = nlp.predecir('¿A qué hora son las clases?')
    assert r['intencion'] == 'horarios' and r['decision'] == 'aceptada'
    assert nlp.predecir('zzzz')['decision'] == 'sin_vocabulario'


def test_preprocesamiento():
    assert nlp.preprocesar('¿Dónde está la escuela?') == ['esta', 'escuela']
