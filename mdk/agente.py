"""Agente inteligente HÍBRIDO de MDK Tuchtlán.

Ciclo: PERCIBIR → DECIDIR → ACTUAR.

- Percibe: texto, imagen opcional y contexto de la sesión (memoria corta).
- Decide con reglas reactivas para lo simple y con pasos deliberativos cuando
  falta información o hay incertidumbre (pedir foto, pedir otra foto, aclarar
  una contradicción, usar la memoria de la conversación).
- Actúa: llama al chatbot de reglas, al SVM de intenciones, a la CNN o deriva
  a atención humana, y devuelve una respuesta combinada con su explicación.

Tabla de decisiones (se documenta igual en el reporte):

  R0  Imagen recibida pero la CNN no está disponible     → reglas + aviso        (seguridad)
  R1  Texto con intención reconocida por reglas          → chatbot de reglas     (reactiva)
  R2  Reglas «desconocida» y SVM con margen suficiente    → SVM + resp. controlada(reactiva)
  R3  Reglas «desconocida» y SVM ambiguo/sin vocabulario  → derivación humana     (reactiva)
  R4  Se refiere a «este/eso» sin foto ni artículo previo → PEDIR FOTO y recordar
                                                            la pregunta           (deliberativa)
  R5  Foto con confianza ≥ 0.65                          → CNN + respuesta según
                                                            la pregunta           (reactiva)
  R6  Foto con confianza < 0.65                          → pedir otra foto       (deliberativa)
  R7  Foto y texto se contradicen                        → aclarar               (deliberativa)
  R8  Seguimiento sin foto sobre el artículo anterior    → usar memoria          (deliberativa)
  R9  Talla, autenticidad, estado u homologación         → declarar el límite    (seguridad)

La CNN identifica el TIPO de artículo. Nunca afirma talla, autenticidad,
estado ni capacidad de protección.
"""
import re

from .chatbot import ChatbotMDK, normalizar, articulo_mencionado, respuesta_equipo, CON_ARTICULO, INDEFINIDO
from .conocimiento import CONTACTO
from .vision import CLASES, ImagenInvalida

UMBRAL_VISUAL = 0.65         # por debajo se pide otra foto (el mismo umbral del avance 2)
UMBRAL_ALTO = 0.80           # por encima se considera confianza alta en la interfaz

NOMBRES = {'careta': 'careta (casco)', 'guantes': 'guantes', 'peto': 'peto', 'uniforme': 'uniforme'}

# Intenciones de reglas que no impiden interpretar la pregunta como referida a un artículo.
GENERICAS = {'desconocida', 'visita', 'costos', 'ayuda', 'contacto', 'saludo'}

DEICTICO = re.compile(
    r'\b(este|esto|estos|ese|eso|esos|aquel)\b(?!\s+(mes|ano|lunes|martes|miercoles|jueves|viernes|'
    r'dia|fin|horario|turno|grupo|precio|costo|sabado|domingo))'
    r'|\b(la|una|mi|esta|esa) (foto|imagen|fotografia)\b'
    r'|\blo tengo\b|\btengo uno\b|\bme regalaron\b|\blo puedo (llevar|usar)\b')

SEGUIMIENTO = re.compile(
    r'\b(cuanto|precio|costo|cuesta|cuestan|vale|talla|tallas|medida|color|colores|sirve|usar|llevar|'
    r'comprar|incluye|lo|la|y)\b')

LIMITES = re.compile(r'\b(talla|tallas|medida|medidas|original|autentic\w*|oficial|homolog\w*|'
                     r'estado|roto|rota|seguro|segura|calidad|marca)\b')

AVISO_LIMITE = ('Recuerda: este asistente no puede comprobar talla, autenticidad, estado ni '
                'homologación de un artículo, ni siquiera con una foto; eso lo confirma la escuela.')

PREGUNTA_FOTO_POR_OMISION = '¿Qué artículo es?'

CANONICAS = {   # pregunta que se envía al chatbot de reglas cuando el SVM decide la intención
    'horarios': 'horarios', 'costos': 'costos', 'inscripcion': 'inscripcion',
    'ubicacion': 'ubicacion', 'uniforme': 'uniforme', 'proteccion': 'equipo de proteccion',
}
TEMAS = {'horarios': 'horarios', 'costos': 'costos', 'inscripcion': 'inscripción',
         'ubicacion': 'ubicación', 'uniforme': 'uniforme', 'proteccion': 'equipo de protección'}

CLAVES_CONTEXTO = ('tipo_alumno', 'mes_consulta', 'cerrado', 'ultima_respuesta',
                   'ultimo_articulo', 'pregunta_pendiente')


def sanitizar_contexto(contexto):
    """El contexto llega del cliente: se aceptan solo claves y valores esperados."""
    if not isinstance(contexto, dict):
        return {}
    limpio = {}
    if contexto.get('tipo_alumno') in ('nuevo', 'reingreso'):
        limpio['tipo_alumno'] = contexto['tipo_alumno']
    mes = contexto.get('mes_consulta')
    if isinstance(mes, int) and not isinstance(mes, bool) and 1 <= mes <= 12:
        limpio['mes_consulta'] = mes
    limpio['cerrado'] = contexto.get('cerrado') is True
    for clave, maximo in (('ultima_respuesta', 2000), ('pregunta_pendiente', 300)):
        valor = contexto.get(clave)
        if isinstance(valor, str) and valor.strip():
            limpio[clave] = valor[:maximo]
    if contexto.get('ultimo_articulo') in ('careta', 'guantes', 'peto', 'uniforme', 'codera',
                                           'espinillera', 'concha', 'empeineras', 'bucal', 'mochila'):
        limpio['ultimo_articulo'] = contexto['ultimo_articulo']
    return limpio


def _articulo_de_clase(clase):
    return 'careta' if clase == 'casco' else clase


class AgenteMDK:
    def __init__(self, nlp, vision, fecha=None):
        self.nlp = nlp            # ClasificadorIntenciones
        self.vision = vision      # ClasificadorVisual
        self.fecha = fecha        # fija la fecha en pruebas; None = hoy en CDMX

    # ------------------------------------------------------------------ API pública
    def atender(self, mensaje='', imagen_bytes=None, contexto=None):
        ctx = sanitizar_contexto(contexto)
        texto = (mensaje or '').strip()[:500]
        plan = []                                         # pasos seguidos (trazabilidad)

        # ---------- PERCIBIR ----------
        if not texto and not imagen_bytes:
            return self._salida('R1', 'reactiva', 'Escribe una pregunta o adjunta una foto para comenzar.',
                                ['entrada_vacia'], None, 'No llegó texto ni imagen.',
                                'Escribe tu pregunta.', ctx, ctx_extra={})
        if not texto and imagen_bytes and ctx.get('pregunta_pendiente'):
            texto = ctx['pregunta_pendiente']
            plan.append('retoma_pregunta_pendiente')
        elif not texto:
            texto = PREGUNTA_FOTO_POR_OMISION
        t = normalizar(texto)
        articulo_texto = articulo_mencionado(t)
        deictico = bool(DEICTICO.search(t))

        bot = ChatbotMDK.desde_contexto(ctx, fecha=self.fecha)
        reglas = bot.responder(texto)
        intenciones = reglas['intenciones']
        generica = set(intenciones) <= GENERICAS
        ctx_bot = bot.exportar_contexto()
        percepcion = {'texto': texto, 'intenciones_reglas': intenciones,
                      'articulo_en_texto': articulo_texto, 'referencia_a_objeto': deictico,
                      'hay_imagen': bool(imagen_bytes), 'memoria_articulo': ctx.get('ultimo_articulo')}

        # ---------- DECIDIR Y ACTUAR ----------
        if imagen_bytes:
            salida = self._con_imagen(texto, imagen_bytes, reglas, articulo_texto, deictico, generica,
                                      ctx, ctx_bot, plan)
        else:
            salida = self._solo_texto(texto, t, reglas, articulo_texto, deictico, generica,
                                      ctx, ctx_bot, plan)

        # R9: regla de seguridad transversal
        sobre_articulo = salida['decision'] in ('R5', 'R8') or (salida['decision'] == 'R1' and articulo_texto)
        if LIMITES.search(t) and sobre_articulo and 'No puedo comprobar' not in salida['respuesta']:
            salida['respuesta'] += '\n\n' + AVISO_LIMITE
            salida['ruta'].append('regla_seguridad_R9')
            salida['reglas_aplicadas'].append('R9')
        salida['percepcion'] = percepcion
        salida['plan'] = plan + salida.pop('_pasos', [])
        return salida

    # ------------------------------------------------------------------ con foto
    def _con_imagen(self, texto, imagen_bytes, reglas, articulo_texto, deictico, generica, ctx, ctx_bot, plan):
        if not self.vision.disponible:
            base = reglas['respuesta'] if reglas['intenciones'] != ['desconocida'] else ''
            respuesta = (base + '\n\n' if base else '') + (
                'En este momento no pude analizar la fotografía. Puedes describir el artículo con '
                'palabras (por ejemplo, «peto» o «careta») o intentarlo más tarde. ' + CONTACTO)
            return self._salida('R0', 'seguridad', respuesta, ['reglas', 'vision_no_disponible'], None,
                                'El clasificador visual no está cargado; se respondió solo con texto.',
                                'Describe el artículo con palabras o vuelve a intentarlo.', ctx_bot,
                                ctx_extra={'ultimo_articulo': ctx.get('ultimo_articulo'),
                                           'pregunta_pendiente': None})

        vision = self.vision.clasificar(imagen_bytes)      # puede lanzar ImagenInvalida
        clase, confianza = vision['clase'], vision['confianza']
        reconocido = _articulo_de_clase(clase)
        nombre = NOMBRES[reconocido]

        if confianza < UMBRAL_VISUAL:                                           # R6
            top = sorted(vision['probabilidades'].items(), key=lambda kv: -kv[1])[:2]
            opciones = ' o '.join(NOMBRES[_articulo_de_clase(c)] for c, _ in top)
            respuesta = (f'No distingo con suficiente claridad el artículo (podría ser {opciones}, '
                         f'confianza {confianza:.0%}). ¿Puedes enviar otra foto con buena luz, '
                         'fondo liso y solo el artículo? Guardé tu pregunta para responderla con la nueva foto.')
            return self._salida('R6', 'deliberativa', respuesta, ['vision', 'pedir_otra_foto'], confianza,
                                'La confianza de la CNN es menor que el umbral de 0.65.',
                                'Enviar otra fotografía.', ctx_bot, vision=vision,
                                ctx_extra={'ultimo_articulo': ctx.get('ultimo_articulo'),
                                           'pregunta_pendiente': texto},
                                pasos=['clasificar_imagen', 'confianza_baja', 'guardar_pregunta'])

        if articulo_texto and articulo_texto != reconocido:                     # R7
            fuera = articulo_texto not in NOMBRES
            extra = (' El clasificador solo reconoce careta, guantes, peto y uniforme.' if fuera else '')
            respuesta = (f'Mencionas {CON_ARTICULO.get(articulo_texto, articulo_texto)}, pero la foto parece mostrar '
                         f'{INDEFINIDO[reconocido]} '
                         f'(confianza {confianza:.0%}).{extra} ¿Me confirmas de qué artículo se trata? '
                         'Si es otro, puedes enviar una foto donde se vea mejor.')
            return self._salida('R7', 'deliberativa', respuesta, ['reglas', 'vision', 'aclaracion'], confianza,
                                'El artículo escrito y el reconocido en la foto no coinciden.',
                                'Confirmar el artículo.', ctx_bot, vision=vision,
                                ctx_extra={'ultimo_articulo': reconocido, 'pregunta_pendiente': None},
                                pasos=['clasificar_imagen', 'detectar_contradiccion'])

        sobre_articulo = generica or 'equipamiento' in reglas['intenciones'] or deictico or articulo_texto
        if sobre_articulo:                                                      # R5
            respuesta = respuesta_equipo(texto, reconocido, foto=True)
            ruta = ['vision', 'conocimiento_equipo']
        else:                                                                   # R5 combinada
            respuesta = (reglas['respuesta'] + f'\n\nSobre la foto: parece haber {INDEFINIDO[reconocido]}. '
                         'El uniforme y el equipo oficial se adquieren exclusivamente en la escuela.')
            ruta = ['reglas', 'vision', 'respuesta_combinada']
        nivel = 'alta' if confianza >= UMBRAL_ALTO else 'media'
        return self._salida('R5', 'reactiva', respuesta, ruta, confianza,
                            f'La CNN reconoció «{clase}» con confianza {nivel} ({confianza:.0%}) y se '
                            'combinó con la pregunta del usuario.',
                            'Puedes preguntar precio, tallas o cuándo se necesita.', ctx_bot, vision=vision,
                            ctx_extra={'ultimo_articulo': reconocido, 'pregunta_pendiente': None},
                            pasos=['clasificar_imagen', 'combinar_con_pregunta'])

    # ------------------------------------------------------------------ solo texto
    def _solo_texto(self, texto, t, reglas, articulo_texto, deictico, generica, ctx, ctx_bot, plan):
        memoria = ctx.get('ultimo_articulo')

        if articulo_texto:                                                      # R1 (equipo)
            return self._salida('R1', 'reactiva', reglas['respuesta'], ['reglas'], None,
                                f'Las reglas reconocieron el artículo «{articulo_texto}» en el texto.',
                                'Puedes enviar una foto si quieres que lo identifique.', ctx_bot,
                                ctx_extra={'ultimo_articulo': articulo_texto, 'pregunta_pendiente': None})

        if deictico and generica and not memoria:                               # R4
            respuesta = ('Para ayudarte con ese artículo necesito verlo. ¿Puedes adjuntar una foto? '
                         'Reconozco careta (casco), guantes, peto y uniforme. Guardé tu pregunta y la '
                         'responderé en cuanto llegue la imagen.')
            return self._salida('R4', 'deliberativa', respuesta, ['pedir_foto'], None,
                                'La pregunta se refiere a un objeto («este», «eso») que no se ha mostrado.',
                                'Adjuntar una foto del artículo.', ctx_bot,
                                ctx_extra={'ultimo_articulo': None, 'pregunta_pendiente': texto},
                                pasos=['detectar_falta_de_informacion', 'guardar_pregunta'])

        if memoria and generica and (deictico or SEGUIMIENTO.search(t)):         # R8
            base = respuesta_equipo(texto, memoria)
            respuesta = f'Sobre {CON_ARTICULO.get(memoria, memoria)}: ' + base[0].lower() + base[1:]
            return self._salida('R8', 'deliberativa', respuesta, ['memoria', 'conocimiento_equipo'], None,
                                f'Pregunta de seguimiento: se usó el artículo recordado («{memoria}»).',
                                'Puedes enviar otra foto para consultar un artículo distinto.', ctx_bot,
                                ctx_extra={'ultimo_articulo': memoria, 'pregunta_pendiente': None},
                                pasos=['recuperar_memoria'])

        if reglas['intenciones'] == ['desconocida']:
            nlp = self.nlp.predecir(texto)
            if nlp['decision'] == 'aceptada':                                   # R2
                intencion = nlp['intencion']
                base = ChatbotMDK(fecha=self.fecha).responder(CANONICAS[intencion])['respuesta']
                respuesta = f'Entiendo que preguntas por {TEMAS[intencion]}. ' + base
                return self._salida('R2', 'reactiva', respuesta, ['reglas', 'svm', 'respuesta_controlada'],
                                    nlp['margen'],
                                    f'Las reglas no reconocieron la intención; el SVM propuso «{intencion}» '
                                    f'con margen {nlp["margen"]}.',
                                    'Si no era tu pregunta, escríbela con otras palabras.', ctx_bot, nlp=nlp,
                                    ctx_extra={'ultimo_articulo': memoria, 'pregunta_pendiente': None})
            motivo = ('la pregunta es ambigua para el modelo' if nlp['decision'] == 'ambigua'
                      else 'no reconozco las palabras de la pregunta')
            respuesta = ('No estoy seguro de haber entendido tu pregunta. Puedo ayudarte con horarios, '
                         'costos, inscripción, ubicación, uniforme y equipo de protección. ' + CONTACTO)
            return self._salida('R3', 'reactiva', respuesta, ['reglas', 'svm', 'derivacion_humana'],
                                nlp['margen'] or None, f'Derivación a atención humana: {motivo}.',
                                'Reformular la pregunta o escribir por WhatsApp.', ctx_bot, nlp=nlp,
                                ctx_extra={'ultimo_articulo': memoria, 'pregunta_pendiente': None})

        return self._salida('R1', 'reactiva', reglas['respuesta'], ['reglas'], None,      # R1
                            'Las reglas reconocieron: ' + ', '.join(reglas['intenciones']) + '.',
                            None, ctx_bot,
                            ctx_extra={'ultimo_articulo': memoria, 'pregunta_pendiente': None})

    # ------------------------------------------------------------------ formato común
    @staticmethod
    def _salida(decision, tipo, respuesta, ruta, confianza, explicacion, siguiente_paso, ctx_bot,
                vision=None, nlp=None, ctx_extra=None, pasos=None):
        contexto = {k: ctx_bot.get(k) for k in ('tipo_alumno', 'mes_consulta', 'cerrado')}
        contexto.update(ctx_extra or {})
        contexto['ultima_respuesta'] = respuesta
        contexto = {k: v for k, v in contexto.items() if k in CLAVES_CONTEXTO and v not in (None, False)}
        return {
            'respuesta': respuesta,
            'decision': decision,
            'tipo_decision': tipo,
            'reglas_aplicadas': [decision],
            'ruta': list(ruta),
            'confianza': confianza,
            'explicacion': explicacion,
            'siguiente_paso': siguiente_paso,
            'vision': vision,
            'nlp': nlp,
            'contexto': contexto,
            '_pasos': pasos or [],
        }


__all__ = ['AgenteMDK', 'sanitizar_contexto', 'ImagenInvalida', 'UMBRAL_VISUAL', 'CLASES']
