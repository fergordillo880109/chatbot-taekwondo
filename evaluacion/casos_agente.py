"""Matriz de pruebas funcionales del agente.

Cada caso es una conversación (uno o varios turnos). En cada turno se indica la
imagen (clave de ``simulacion.COLORES`` en las pruebas automáticas; en el
notebook se sustituye por una fotografía real) y la decisión esperada.
"""
CASOS = [
    ('C01', 'Horario por reglas', [('¿Qué días hay clases?', None, 'R1')]),
    ('C02', 'Costo por reglas', [('¿Cuánto cuesta la mensualidad?', None, 'R1')]),
    ('C03', 'Saludo', [('hola', None, 'R1')]),
    ('C04', 'Fin de semana («este sábado» no es un objeto)', [('¿Hay clases este sábado?', None, 'R1')]),
    ('C05', 'Edad con seguimiento de contexto',
     [('¿Desde qué edad aceptan niños?', None, 'R1'), ('7', None, 'R1')]),
    ('C06', 'Promoción de nuevo ingreso',
     [('Quiero inscribir a mi hijo en noviembre, ¿hay promoción?', None, 'R1')]),
    ('C07', 'Reglas no entienden → SVM', [('Me puedes dar referencias del lugar porque no lo encuentro', None, 'R2')]),
    ('C08', 'Reglas no entienden → SVM', [('¿Hay lugares disponibles para nuevos alumnos?', None, 'R2')]),
    ('C09', 'Sin vocabulario → humano', [('asdf qwerty', None, 'R3')]),
    ('C10', 'Pide foto y retoma la pregunta',
     [('Tengo este, ¿lo puedo llevar?', None, 'R4'), ('', 'casco_claro', 'R5')]),
    ('C11', 'Foto + precio y seguimiento con memoria',
     [('¿Cuánto cuesta?', 'peto_claro', 'R5'), ('¿y qué tallas hay?', None, 'R8')]),
    ('C12', 'Foto dudosa → otra foto → respuesta',
     [('¿Qué es esto?', 'dudosa', 'R6'), ('', 'guantes_claro', 'R5')]),
    ('C13', 'Texto y foto se contradicen', [('¿Este casco sirve para entrenar?', 'peto_claro', 'R7')]),
    ('C14', 'Pregunta no relacionada + foto (respuesta combinada)', [('¿Dónde están?', 'uniforme_claro', 'R5')]),
    ('C15', 'Foto sin texto', [('', 'uniforme_claro', 'R5')]),
    ('C16', 'Autenticidad (regla de seguridad)', [('¿El peto que tengo es original?', None, 'R1')]),
    ('C17', 'Talla con foto', [('¿Qué talla es?', 'casco_claro', 'R5')]),
    ('C18', 'Artículo en texto, luego seguimiento', [('¿Cuánto cuestan los guantes?', None, 'R1'),
                                                     ('¿y qué colores hay?', None, 'R8')]),
]


def ejecutar(agente, obtener_imagen):
    """Ejecuta todos los casos. ``obtener_imagen(clave)`` devuelve bytes o None."""
    filas = []
    for codigo, descripcion, turnos in CASOS:
        contexto = {}
        for i, (texto, clave_imagen, esperada) in enumerate(turnos):
            imagen = obtener_imagen(clave_imagen) if clave_imagen else None
            r = agente.atender(texto, imagen, contexto)
            contexto = r['contexto']
            filas.append({
                'caso': codigo, 'turno': i + 1, 'descripcion': descripcion,
                'entrada': texto or '(sin texto)', 'imagen': clave_imagen or '—',
                'esperada': esperada, 'obtenida': r['decision'],
                'reglas': '+'.join(r['reglas_aplicadas']), 'ruta': ' → '.join(r['ruta']),
                'confianza': r['confianza'], 'correcta': r['decision'] == esperada,
                'respuesta': r['respuesta'],
            })
    return filas
