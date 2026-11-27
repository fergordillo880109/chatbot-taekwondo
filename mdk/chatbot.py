r"""Chatbot basado en reglas de MDK Tuchtlán (Proyectos 1 y 2).

Se conserva la lógica original. Cambios para el proyecto final:
- El estado de la conversación se puede exportar/importar como diccionario
  (``exportar_contexto`` / ``desde_contexto``). Así la API Flask no guarda
  nada entre solicitudes: el contexto de cada usuario vive en su sesión de
  Streamlit y se envía con cada mensaje.
- La fecha por omisión es la de la Ciudad de México, no la del servidor.
- NLTK es opcional: si no está instalado se usa una expresión regular
  equivalente a ``RegexpTokenizer(r'\w+')``.
"""
import re
import unicodedata
from datetime import date, datetime
from zoneinfo import ZoneInfo

from .conocimiento import ESCUELA, CONTACTO

try:
    from nltk.tokenize import RegexpTokenizer
    TOKENIZADOR = RegexpTokenizer(r'\w+')
    _tokenizar = TOKENIZADOR.tokenize
except ImportError:  # pragma: no cover - depende del entorno
    def _tokenizar(texto):
        return re.findall(r'\w+', texto)

ZONA = ZoneInfo('America/Mexico_City')


def hoy_mexico():
    return datetime.now(ZONA).date()

# MÓDULO 3 — Preparación del mensaje con NLTK

def normalizar(texto):
    """NLTK separa tokens sin descargar corpus; conserva números y palabras."""
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto.lower())
                    if unicodedata.category(c) != 'Mn')
    return ' '.join(_tokenizar(texto))

# MÓDULO 4 — Reglas de reconocimiento de intenciones
# Se permiten varias intenciones en un mensaje; orden fijo y respuestas sin azar.
PATRONES = {
    'reingreso': r'\b(reingreso|regresar|regreso|regresa|volver|baja|exalumno)\b',
    'promocion': r'\b(promocion|promociones|descuento|gratis|gratuito|gratuita|2x1|octubre|noviembre|diciembre|enero)\b',
    'ubicacion': r'\b(donde|ubicacion|direccion|domicilio)\b|como (?:puedo |podemos )?llegar\b',
    'contacto': r'\b(whatsapp|telefono|contacto|numero|comunicar|humano|persona)\b',
    'horarios': r'\b(horario|horarios|dias|hora|turno|turnos|sabado|sabados|domingo|domingos|viernes)\b',
    'visita': r'\b(?:puedo|podemos|podria|podriamos) (?:ir|asistir|llegar|llevar|visitar)\b|que (?:necesito|debo|tengo que) (?:para )?(?:llevar|asistir|ir)\b|primera clase|primer dia',
    'fin_semana': r'\b(sabado|sabados|domingo|domingos|sabatino|sabatinos|dominical|dominicales)\b|fin(?:es)? de semana|weekend',
    'edades': r'\b(edad|edades|anos|preescolar|preescolares|adultos|adulto|nina|ninas|nino|ninos|juvenil|juveniles|infantil|infantiles)\b',
    'documentos': r'\b(documentos|documentacion|requisitos|acta|ine|fotografias|papeles)\b|que (?:necesito|necesita|necesitan|piden|se necesita|debo llevar|tengo que llevar) para (?:inscribirme|inscribirse|inscribir|ingresar|entrar)\b',
    'clase_muestra': r'\b(prueba|muestra|probar|cita|reservar|reserva)\b',
    'uniforme': r'\b(uniforme|dobok|ropa|pants|playera|logos|logotipos)\b',
    'proteccion': r'\b(proteccion|protecciones|equipo|amarilla|careta|casco|peto|petos|guantes|codera|coderas|espinillera|espinilleras|concha|bucal|empeineras|mochila)\b',
    'experiencia': r'\b(principiante|principiantes|avanzado|avanzados|experiencia|constancia)\b|otra escuela|nunca he practicado',
    'evaluaciones': r'\b(examen|examenes|kup|reforzamiento|grado|cinta|cuello)\b',
    'mensualidad': r'\b(mensualidad|colegiatura|mensualidades|mes)\b|al mes',
    'inscripcion': r'\b(inscripcion|inscribirme|inscribirse|inscribir|inscribo|ingresar)\b',
    'costos': r'\b(costo|costos|precio|precios|tarifa|tarifas)\b|cuanto (?:cuesta|cuestan|cobran|vale|valen|pago|pagaria|debo|es|son)',
    'descripcion_clases': r'como (?:son|es|funcionan|funciona) (?:las |la |sus |tus )?clases?\b|que (?:hacen|ensenan|aprenden|se aprende|se hace) (?:en |durante )?(?:las |la |sus )?clases?\b|en que consisten (?:las |sus )?clases\b',
    'identidad': r'que es mdk|que es la escuela|asociacion|moo duk',
    'ayuda': r'\b(ayuda|menu|informes|informacion)\b',
}
REGLAS = {k: re.compile(v) for k, v in PATRONES.items()}

# Respuestas controladas sobre artículos oficiales y precios de la escuela.
def articulo_mencionado(t):
    for patron, articulo in [
        (r'\b(uniforme|dobok)\b', 'uniforme'),
        (r'\b(careta|casco)\b', 'careta'),
        (r'\b(petos?|pechera)\b', 'peto'),
        (r'\bguantes?\b', 'guantes'),
        (r'\bcoderas?\b', 'codera'),
        (r'\bespinilleras?\b', 'espinillera'),
        (r'\bconcha\b', 'concha'),
        (r'\b(empeineras?|empeineras?)\b', 'empeineras'),
        (r'\bprotector bucal|\bbucal\b', 'bucal'),
        (r'\bmochilas?\b', 'mochila')]:
        if re.search(patron, t):
            return articulo
    return None

CON_ARTICULO = {'careta': 'la careta (casco de protección)', 'peto': 'el peto', 'guantes': 'los guantes',
                'uniforme': 'el uniforme', 'codera': 'la codera', 'espinillera': 'la espinillera',
                'concha': 'la concha', 'empeineras': 'las empeineras', 'bucal': 'el protector bucal',
                'mochila': 'la mochila'}
INDEFINIDO = {'careta': 'una careta (casco de protección)', 'peto': 'un peto', 'guantes': 'unos guantes',
              'uniforme': 'un uniforme', 'codera': 'una codera', 'espinillera': 'una espinillera',
              'concha': 'una concha', 'empeineras': 'unas empeineras', 'bucal': 'un protector bucal',
              'mochila': 'una mochila'}


def _de(frase):
    """«de» + «el peto» → «del peto»."""
    return 'del ' + frase[3:] if frase.startswith('el ') else 'de ' + frase


def respuesta_equipo(pregunta, articulo=None, foto=False):
    t = normalizar(pregunta)
    articulo = articulo or articulo_mencionado(t)
    e = ESCUELA['equipo']
    if articulo == 'casco':
        articulo = 'careta'
    nombres = {'careta':'careta (casco de protección)', 'peto':'peto',
               'guantes':'guantes', 'uniforme':'uniforme', 'codera':'codera',
               'espinillera':'espinillera', 'concha':'concha',
               'empeineras':'empeineras', 'bucal':'protector bucal',
               'mochila':'mochila'}
    nombre = nombres.get(articulo, 'equipo de protección')
    con = CON_ARTICULO.get(articulo, 'el equipo de protección')   # «el peto», «los guantes»
    intro = f'En la foto parece haber {INDEFINIDO.get(articulo, nombre)}. ' if foto else ''
    compra = 'El uniforme y todo el equipo de protección oficial se adquieren exclusivamente en la escuela.'
    consulta_precio = bool(re.search(r'\b(cuanto|costo|costos|precio|precios|vale|cuestan|paquete)\b', t))
    consulta_uso = bool(re.search(r'\b(sirve|servira|funcional|funciona|usar|llevar|aceptan|permiten|mio|este|esta)\b', t))
    if articulo == 'uniforme':
        base = 'El uniforme oficial cuesta $1,000 MXN e incluye casaca, pantalón y cinta blanca. '
        if foto and consulta_uso:
            return intro + 'Para las clases se utiliza el uniforme oficial adquirido en la escuela; uno de otro color o con otros logotipos no se permite. Mientras lo compras, puedes asistir con pants y playera.'
        return intro + base + compra + ' Mientras lo compras, puedes asistir con pants y playera.'
    if articulo and (consulta_precio or re.search(r'\btalla|\bmedida|\bdimens|\bcolor',t)):
        precios = {'careta':'$850 MXN; tallas chica, mediana y grande; colores azul o rojo',
                   'peto':'$905 MXN en tallas 00 a 4, o $1,015 MXN en talla 5',
                   'guantes':'$585 MXN por par; tallas chica, mediana y grande',
                   'codera':'$605 MXN por unidad; tallas extrachica, chica, mediana y grande',
                   'espinillera':'$605 MXN por unidad; tallas extrachica, chica, mediana y grande',
                   'concha':'$525 MXN por unidad; modelos para niña o niño y tallas chica, mediana y grande',
                   'empeineras':'$585 MXN por par; tallas chica, mediana y grande',
                   'bucal':'$200 MXN', 'mochila':'$930 MXN'}
        dato = precios.get(articulo, '')
        if re.search(r'\b(talla|medida|dimension)\w*\b',t):
            return intro + f'Las opciones {_de(con)} son: {dato}. No puedo comprobar la talla ni medirlo a partir de la foto; en la escuela pueden ayudarte a elegirlo. ' + compra
        if 'color' in t:
            return intro + (f'La escuela ofrece {con} en estas opciones: {dato}. ' if articulo == 'careta' else f'Para {con}, confirma los colores disponibles en la escuela. ') + 'La foto no basta para determinar si un artículo externo es oficial. ' + compra
        return intro + f'El precio {_de(con)} es {dato}. ' + compra
    if articulo and foto:
        if re.search(r'\b(es|parece|identifica|que es|cual es)\b',t) and not consulta_uso:
            return intro.strip() + ' ' + compra
        if re.search(r'\b(funciona|funcional|estado|roto|seguro)\b',t):
            return intro + 'No puedo comprobar su estado ni su capacidad de protección con una fotografía. ' + compra
        return intro + 'Para entrenar se utiliza el equipo oficial adquirido en la escuela; una foto no permite comprobar su talla ni su estado. ' + compra
    if articulo:
        verbo = 'se adquieren' if articulo in ('guantes', 'empeineras') else 'se adquiere'
        return (f'{con[0].upper() + con[1:]} {verbo} directamente en la escuela: el uniforme y todo el '
                'equipo de protección oficial se venden exclusivamente allí.')
    return ('El equipo básico incluye peto, careta (casco), espinilleras, coderas y concha. '
            f'El paquete completo cuesta ${e["paquete"]:,} MXN; con mochila, ${e["paquete_mochila"]:,} MXN. '
            'Es obligatorio tener el equipo básico completo al llegar a cinta amarilla (9.º kup). '
            'Si el alumno aún es cinta blanca, puede comprarlo antes o comenzar con peto y careta. '
            'Guantes, empeineras, bucal y mochila son adicionales. ' + compra)

# MÓDULO 5 — Motor de conversación y respuestas
class ChatbotMDK:
    def __init__(self, fecha=None):
        # Fecha explícita para reproducir las pruebas. La interfaz usa fecha actual de México.
        self.fecha = fecha or hoy_mexico()
        self.historial = []
        self.tipo_alumno = None
        self.mes_consulta = None
        self.cerrado = False

    def promocion(self, texto):
        c = ESCUELA['costos']
        if self.tipo_alumno == 'reingreso':
            return f"La promoción es exclusiva de nuevo ingreso. En reingreso se paga la colegiatura vigente: ${c['mensualidad']:,} MXN al mes. ¡Nos dará gusto recibirte de nuevo!"
        if self.mes_consulta in (11, 12):
            gratis = 'diciembre de 2026' if self.mes_consulta == 11 else 'enero de 2027'
            uniforme = round(c['uniforme'] * (1-ESCUELA['promocion_nov_dic']['descuento_uniforme']))
            total = c['inscripcion'] + c['mensualidad'] + uniforme
            return (f"Para NUEVO INGRESO en noviembre y diciembre de 2026: inscripción ${c['inscripcion']:,}, primera mensualidad ${c['mensualidad']:,} y la siguiente gratis; uniforme con 25 % de descuento: ${uniforme:,}. Suma de estos conceptos: ${total:,} MXN. Si el primer mes pagado es {'noviembre' if self.mes_consulta == 11 else 'diciembre'}, el mes gratis es {gratis}. Tienes una semana desde la inscripción para comprar el uniforme conservando la promoción. No aplica a reingresos.")
        if self.mes_consulta == 10:
            return '¡En octubre de 2026 tu inscripción es gratis y tienes 20 % de descuento en el uniforme! El uniforme queda en $800 y la mensualidad es de $600. La suma de uniforme y primera mensualidad es $1,400 MXN. Tienes una semana desde tu inscripción para comprar el uniforme conservando la promoción.'
        return '¡Con gusto te orientamos sobre las promociones para la fecha que te interesa! ' + CONTACTO

    def responder(self, mensaje):
        t = normalizar(mensaje)
        if re.fullmatch(r'\d{1,2}(?: anos)?', t) and self.historial and 'que edad' in normalizar(self.historial[-1]['respuesta']):
            t = re.search(r'\d+', t).group() + ' anos'
        if not t:
            return {'intenciones': ['vacio'], 'respuesta': 'Escribe una pregunta para comenzar.'}
        if self.cerrado:
            return {'intenciones': ['cerrado'], 'respuesta': 'La conversación terminó. Pulsa Nueva conversación para comenzar otra.'}
        if re.fullmatch(r'(salir|adios|hasta luego|terminar|cerrar)', t):
            self.cerrado = True
            return self._guardar(mensaje, ['despedida'], 'Gracias por consultar MDK Tuchtlán. ¡Hasta pronto!')
        if re.search(r'\b(reingreso|regresar|regreso|regresa|volver|baja|exalumno)\b', t):
            self.tipo_alumno = 'reingreso'
        elif re.search(r'nuevo ingreso|primera vez|soy nuevo|es nuevo', t):
            self.tipo_alumno = 'nuevo'
        meses = {'octubre':10, 'noviembre':11, 'diciembre':12, 'enero':1}
        encontrados = [v for k,v in meses.items() if re.search(r'\b'+k+r'\b', t)]
        if encontrados:
            self.mes_consulta = encontrados[0]
        elif self.mes_consulta is None:
            self.mes_consulta = self.fecha.month if self.fecha.year == 2026 else None
        # Beneficios o descuentos retoman la promoción vigente, incluso si se pregunta «¿y los beneficios?».
        # Esta consulta se resuelve antes de «mes», que de otro modo activaría la mensualidad.
        if re.search(r'\b(beneficio|beneficios|descuento|descuentos)\b', t):
            anos = re.findall(r'\b20\d{2}\b', t)
            if anos and any(a != '2026' for a in anos) and not ('enero' in t and '2027' in anos):
                respuesta = 'Para conocer las promociones de ese año, ' + CONTACTO[0].lower() + CONTACTO[1:]
            else:
                respuesta = self.promocion(t)
            return self._guardar(mensaje, ['promocion'], respuesta)
        intenciones = [k for k,r in REGLAS.items() if r.search(t)]
        if re.fullmatch(r'(hola|buenas|buenos dias|buenas tardes|buenas noches)',t):
            intenciones = ['saludo']
        # Evitar respuestas genéricas duplicadas cuando hay temas específicos.
        if 'fin_semana' in intenciones:
            intenciones = [i for i in intenciones if i not in ('horarios', 'visita')]
        if 'costos' in intenciones and not re.search(r'cuantos anos|desde que edad|a que edad', t):
            intenciones = [i for i in intenciones if i != 'edades']
        if 'visita' in intenciones and not re.search(r'edad|anos|adulto', t):
            intenciones = [i for i in intenciones if i != 'edades']
        if 'ubicacion' in intenciones and re.search(r'como (?:puedo |podemos )?llegar', t):
            intenciones = [i for i in intenciones if i != 'visita']
        if 'clase_muestra' in intenciones and not re.search(r'promocion|descuento|noviembre|diciembre|2x1',t):
            intenciones = [i for i in intenciones if i != 'promocion']
        if 'uniforme' in intenciones or 'evaluaciones' in intenciones or 'mensualidad' in intenciones or 'proteccion' in intenciones:
            intenciones = [i for i in intenciones if i != 'costos']
        if 'documentos' in intenciones and not re.search(r'costo|precio|cuanto|cobran', t):
            intenciones = [i for i in intenciones if i != 'inscripcion']
        if 'proteccion' in intenciones and not re.search(r'examen|kup|grado|cuello',t):
            intenciones = [i for i in intenciones if i != 'evaluaciones']
        if not intenciones and t in ('y en diciembre','y diciembre','y en noviembre','y noviembre'):
            intenciones = ['promocion']
        if not intenciones:
            intenciones = ['desconocida']
        # El equipamiento tiene prioridad sobre patrones genéricos de costo o visita.
        if ('proteccion' in intenciones or 'uniforme' in intenciones or
            articulo_mencionado(t) or re.search(r'\b(equipo completo|equipo basico)\b',t)):
            return self._guardar(mensaje, ['equipamiento'], respuesta_equipo(mensaje))
        c = ESCUELA['costos']
        respuestas = {
            'saludo': '¡Hola! Bienvenido a MDK Tuchtlán. Con gusto te ayudamos a conocer nuestros horarios, clases, costos y promociones. ¿Qué te gustaría saber?',
            'descripcion_clases': '¡Nuestras clases son de taekwondo y recibimos alumnos principiantes y avanzados! Tenemos grupos preescolares, infantiles y juveniles, y te ayudamos a elegir el grupo según la edad y el nivel. Promovemos la disciplina, el respeto y la constancia. ¡Puedes conocernos con una clase muestra gratuita! ¿Qué edad tiene quien desea asistir?',
            'identidad': '¡Somos MDK Tuchtlán! Una escuela de taekwondo de la Asociación Mexicana MDK, sistema Moo Duk Kwan. Promovemos la disciplina, el respeto y la constancia.',
            'horarios': ESCUELA['horarios'],
            'fin_semana': 'Por el momento, nuestras clases regulares son de lunes a jueves; no tenemos clases regulares los sábados ni domingos. Tenemos dos horarios: de 5:00 a 6:00 p. m. para Preescolar e Infantil Menor, y de 6:00 a 7:00 p. m. para Infantil Mayor y Juveniles. ¡Con gusto te ayudamos a elegir el grupo según la edad y el nivel!',
            'ubicacion': '¡Nos encontramos en '+ESCUELA['direccion']+'. ¡Te esperamos!',
            'contacto': CONTACTO,
            'edades': '¡Sí! Tenemos clases para niñas y niños desde los 3 años, en grupos preescolares, infantiles y juveniles. ¿Qué edad tiene tu hija o hijo? Así te orientamos sobre su grupo.',
            'visita': '¡Nos encantará recibirlos! Tenemos clases de lunes a jueves: de 5:00 a 6:00 p. m. para Preescolar e Infantil Menor, y de 6:00 a 7:00 p. m. para Infantil Mayor y Juveniles. Para conocer las clases pueden asistir a una clase muestra gratuita con pants y playera. Escríbenos por WhatsApp al 961-654-4587 para coordinar su visita y elegir el horario según la edad y el nivel.',
            'documentos': 'Para integrar tu expediente te pedimos copia del acta de nacimiento, copia de la INE del tutor y dos fotografías tamaño infantil. ¡Puedes inscribirte y entregar estos documentos después! Escríbenos por WhatsApp al 961-654-4587 y te ayudamos a elegir el grupo y realizar tu inscripción.',
            'clase_muestra': '¡Sí! Tenemos una clase muestra gratuita para que nos conozcas. Puedes venir con pants y playera. Escríbenos por WhatsApp al 961-654-4587 y con gusto te ayudamos a coordinar tu visita.',
            'uniforme': f"Nuestro uniforme tiene un precio regular de ${c['uniforme']:,} MXN y se adquiere aquí en la escuela. Al inscribirte tienes una semana para comprarlo conservando la promoción aplicable; mientras tanto puedes asistir con pants y playera. Usamos el dobok oficial: no se permite uno de otro color o con otros logotipos. ¡Pregunta por la promoción del mes en que te inscribirás!",
            'proteccion': 'Para entrenar, el equipo de protección es obligatorio a partir de cinta amarilla y se adquiere en nuestra escuela. Escríbenos por WhatsApp al 961-654-4587 y te ayudamos con los precios y el equipo que necesitas.',
            'experiencia': '¡Claro! Recibimos alumnos principiantes y avanzados. Si ya practicaste taekwondo y tienes constancia de tu último grado adquirido, ¡te lo respetamos!',
            'evaluaciones': 'Nuestros costos de evaluación son: examen de cambio de grado $750 MXN; cinta Kup $135; cuello Kup $90; y examen de reforzamiento para preescolares $300. Para conocer las próximas fechas y lo que corresponde a tu grado, escríbenos por WhatsApp al 961-654-4587.',
            'mensualidad': f"Nuestra mensualidad regular es de ${c['mensualidad']:,} MXN. Los exámenes y artículos Kup tienen un costo aparte. ¡También puedes preguntarnos por las promociones de nuevo ingreso!",
            'inscripcion': f"¡Nos encantará recibirte! La inscripción regular cuesta ${c['inscripcion']:,} MXN. Te ayudamos a elegir el grupo según la edad y el nivel del alumno. Escríbenos por WhatsApp al 961-654-4587 para acompañarte con tu inscripción y conocer la promoción aplicable.",
            'costos': '¡Claro! Nuestras clases tienen una mensualidad de $600 MXN. La inscripción regular cuesta $300 y el uniforme $1,000. La suma de estos tres conceptos es $1,900 antes de promociones. ¡Pregunta por los beneficios del mes en que quieres ingresar!',
            'ayuda': '¡Con gusto te ayudamos! Puedes preguntar por horarios, edades, dirección, costos, documentos, clase muestra, uniforme, equipo de protección y promociones. ¿Por dónde te gustaría empezar?',
            'desconocida': '¡Con gusto te ayudamos! ¿Tu consulta es sobre horarios, costos, inscripción o clases? También puedes escribirnos por WhatsApp al 961-654-4587 para recibir atención personal.',
        }
        if 'edades' in intenciones:
            if re.search(r'\badultos?\b', t):
                respuestas['edades'] = '¡Con gusto te orientamos sobre las opciones para adultos! '+CONTACTO
            else:
                edad_match = re.search(r'\b(\d+) anos\b', t)
                if edad_match and int(edad_match.group(1)) < 3:
                    respuestas['edades'] = 'Si tu pequeño aún no cumple 3 años, escríbenos por WhatsApp al 961-654-4587 para valorar su integración mediante una clase de prueba. ¡Con gusto te orientamos!'
                elif edad_match:
                    edad = int(edad_match.group(1))
                    if 3 <= edad < 9:
                        respuestas['edades'] = f'¡Para {edad} años, el grupo de Preescolar e Infantil Menor puede ser una buena opción! Sus clases son de lunes a jueves, de 5:00 a 6:00 p. m. Elegimos el grupo según la edad y el nivel. ¡Pueden conocernos con una clase muestra gratuita!'
                    elif edad == 9:
                        respuestas['edades'] = '¡A los 9 años valoramos el grupo según su edad y nivel! Tenemos el turno de 5:00 a 6:00 p. m. y el de 6:00 a 7:00 p. m., de lunes a jueves. Escríbenos por WhatsApp al 961-654-4587 para orientarte.'
                    elif edad < 18:
                        respuestas['edades'] = '¡Tenemos el grupo de Infantil Mayor y Juveniles de lunes a jueves, de 6:00 a 7:00 p. m.! Te ayudamos a elegir el grupo según la edad y el nivel. ¡Puedes conocernos con una clase muestra gratuita!'
                    else:
                        respuestas['edades'] = '¡Con gusto te orientamos sobre las opciones para adultos! '+CONTACTO
        if 'reingreso' in intenciones:
            respuestas['reingreso'] = 'Si te diste de baja y regresas, pagas la colegiatura vigente: $600 MXN al mes. La promoción de nuevo ingreso no aplica a reingresos.'
        if 'promocion' in intenciones:
            # Año ajeno no activa promociones de 2026.
            anos = re.findall(r'\b20\d{2}\b', t)
            if anos and any(a != '2026' for a in anos) and not ('enero' in t and '2027' in anos):
                respuestas['promocion'] = 'Para conocer las promociones de ese año, '+CONTACTO[0].lower()+CONTACTO[1:]
            else:
                respuestas['promocion'] = self.promocion(t)
        salida = '\n\n'.join(dict.fromkeys(respuestas[i] for i in intenciones))
        return self._guardar(mensaje, intenciones, salida)

    # ---------------- contexto serializable (proyecto final) ----------------
    def exportar_contexto(self):
        """Estado mínimo para continuar la conversación en la siguiente solicitud."""
        return {
            'tipo_alumno': self.tipo_alumno,
            'mes_consulta': self.mes_consulta,
            'cerrado': self.cerrado,
            'ultima_respuesta': self.historial[-1]['respuesta'] if self.historial else None,
        }

    @classmethod
    def desde_contexto(cls, contexto=None, fecha=None):
        bot = cls(fecha=fecha)
        contexto = contexto or {}
        if contexto.get('tipo_alumno') in ('nuevo', 'reingreso'):
            bot.tipo_alumno = contexto['tipo_alumno']
        mes = contexto.get('mes_consulta')
        if isinstance(mes, int) and 1 <= mes <= 12:
            bot.mes_consulta = mes
        bot.cerrado = bool(contexto.get('cerrado', False))
        ultima = contexto.get('ultima_respuesta')
        if isinstance(ultima, str) and ultima:
            # Solo se necesita para el seguimiento «¿qué edad tiene?» → «7».
            bot.historial.append({'turno': 0, 'entrada': '', 'intenciones': [],
                                  'respuesta': ultima[:2000],
                                  'fecha_consulta': bot.fecha.isoformat()})
        return bot

    def _guardar(self, mensaje, intenciones, respuesta):
        fila = {'turno': len(self.historial)+1, 'entrada': mensaje,
                'intenciones': intenciones, 'respuesta': respuesta,
                'fecha_consulta': self.fecha.isoformat()}
        self.historial.append(fila)
        return fila

