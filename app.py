"""Interfaz web (Streamlit) del asistente MDK Tuchtlán.

La interfaz NO carga modelos: envía texto e imágenes a la API Flask y muestra
la respuesta del agente. Si la API no está corriendo en esta máquina, la inicia
en segundo plano (así funciona igual en Colab y en Streamlit Community Cloud).

Ejecutar:  streamlit run app.py
"""
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

RAIZ = Path(__file__).resolve().parent
API_URL = os.getenv('API_URL', 'http://127.0.0.1:5000').rstrip('/')
WHATSAPP = 'https://wa.me/529616544587'
EJEMPLOS = ['¿Qué días hay clases?', '¿Cuánto cuesta la mensualidad?', '¿Hay promociones este mes?',
            'Tengo este, ¿lo puedo llevar?', '¿Dónde están?']
NOMBRE_CLASE = {'casco': 'Careta (casco)', 'guantes': 'Guantes', 'peto': 'Peto', 'uniforme': 'Uniforme'}

st.set_page_config(page_title='Asistente MDK Tuchtlán', page_icon='🥋', layout='wide')


# ======================================================================= conexión con la API
def api_viva(timeout=2):
    try:
        return requests.get(f'{API_URL}/health', timeout=timeout).ok
    except requests.exceptions.RequestException:
        return False


@st.cache_resource(show_spinner=False)
def iniciar_api():
    """Arranca la API una sola vez por servidor (recurso compartido por todos los usuarios)."""
    if api_viva():
        return 'externa'
    if os.getenv('AUTO_API', '1') == '0' or not API_URL.startswith(('http://127.0.0.1', 'http://localhost')):
        return 'no_disponible'
    (RAIZ / 'logs').mkdir(exist_ok=True)
    try:
        import gunicorn  # noqa: F401
        comando = [sys.executable, '-m', 'gunicorn', 'api:app', '--workers', '1', '--threads', '4',
                   '--timeout', '120', '--bind', '127.0.0.1:5000']
    except ImportError:
        comando = [sys.executable, 'api.py']
    proceso = subprocess.Popen(comando, cwd=RAIZ, stdout=open(RAIZ / 'logs' / 'api_proceso.log', 'a'),
                               stderr=subprocess.STDOUT)
    for _ in range(240):                       # TensorFlow puede tardar en cargar
        if api_viva():
            return 'iniciada'
        if proceso.poll() is not None:
            return 'fallo'
        time.sleep(1)
    return 'tardando'


def llamar_api(ruta, **kwargs):
    """Devuelve (datos, error) sin romper la interfaz."""
    try:
        r = requests.post(f'{API_URL}{ruta}', timeout=60, **kwargs)
        datos = r.json()
        return (datos, None) if r.ok else (None, datos.get('error', 'Ocurrió un error inesperado.'))
    except requests.exceptions.Timeout:
        return None, 'El servicio tardó demasiado en responder. Intenta de nuevo.'
    except (requests.exceptions.RequestException, ValueError):
        return None, 'El servicio de IA no está disponible en este momento. Intenta en unos minutos.'


@st.cache_data(ttl=10, show_spinner=False)
def estado_api():
    try:
        return requests.get(f'{API_URL}/health', timeout=3).json()
    except (requests.exceptions.RequestException, ValueError):
        return None


def miniatura(datos):
    """Copia reducida para mostrar en el historial; la foto original no se guarda."""
    imagen = Image.open(io.BytesIO(datos)).convert('RGB')
    imagen.thumbnail((320, 320))
    buffer = io.BytesIO()
    imagen.save(buffer, format='JPEG', quality=80)
    return buffer.getvalue()


# ======================================================================= estado de sesión (por usuario)
for clave, valor in {'mensajes': [], 'contexto': {}, 'prediccion': None, 'ejemplo': None}.items():
    st.session_state.setdefault(clave, valor)


def nueva_conversacion():
    st.session_state.mensajes = []
    st.session_state.contexto = {}
    st.session_state.prediccion = None


def enviar_al_agente(texto, archivo=None):
    st.session_state.mensajes.append({'rol': 'user', 'texto': texto,
                                      'imagen': miniatura(archivo.getvalue()) if archivo else None})
    datos = {'mensaje': texto, 'contexto': json.dumps(st.session_state.contexto, ensure_ascii=False)}
    files = {'file': (archivo.name, archivo.getvalue(), archivo.type)} if archivo else None
    with st.spinner('El asistente está analizando tu consulta…'):
        r, error = llamar_api('/agente', data=datos, files=files)
    if error:
        st.session_state.mensajes.append({'rol': 'assistant', 'texto': error, 'error': True})
    else:
        st.session_state.contexto = r['contexto']
        st.session_state.mensajes.append({'rol': 'assistant', 'texto': r['respuesta'], 'detalle': r})


def mostrar_detalle(d):
    with st.expander('¿Cómo decidió el asistente?'):
        tipo = {'reactiva': 'reactiva (regla directa)', 'deliberativa': 'deliberativa (planeó un paso extra)',
                'seguridad': 'de seguridad'}.get(d['tipo_decision'], d['tipo_decision'])
        st.markdown(f"**Regla:** {' + '.join(d['reglas_aplicadas'])} · decisión {tipo}  \n"
                    f"**Componentes usados:** {' → '.join(d['ruta'])}  \n"
                    f"**Explicación:** {d['explicacion']}")
        if d.get('vision'):
            v = d['vision']
            st.caption(f"Clasificador visual: {NOMBRE_CLASE[v['clase']]} · confianza {v['confianza']:.0%}")
            st.bar_chart(pd.Series(v['probabilidades']).rename(index=NOMBRE_CLASE), height=160)
        if d.get('nlp'):
            st.caption(f"Clasificador de intenciones: {d['nlp']['intencion']} · margen {d['nlp']['margen']}")
        if d.get('siguiente_paso'):
            st.caption(f"Siguiente paso sugerido: {d['siguiente_paso']}")


# ======================================================================= arranque
with st.spinner('Iniciando los servicios de IA (la primera vez puede tardar hasta un minuto)…'):
    arranque = iniciar_api()
salud = estado_api()

# ======================================================================= barra lateral
with st.sidebar:
    st.header('🥋 MDK Tuchtlán')
    if salud is None:
        st.error('Servicio de IA fuera de línea')
    elif salud['estado'] == 'ok':
        st.success('Servicio de IA en línea')
    else:
        st.warning('Servicio parcial: el reconocimiento de fotos no está disponible')
    st.button('🔄 Nueva conversación', on_click=nueva_conversacion, width='stretch')
    st.subheader('Prueba con')
    for ejemplo in EJEMPLOS:
        if st.button(ejemplo, key=f'ej_{ejemplo}', width='stretch'):
            st.session_state.ejemplo = ejemplo
    st.divider()
    st.caption('Prototipo académico. Precios y promociones deben confirmarse con la escuela.')
    st.link_button('Hablar con una persona (WhatsApp)', WHATSAPP, width='stretch')

# ======================================================================= contenido
st.title('Asistente de la escuela de taekwondo MDK Tuchtlán')
st.write('Resuelve dudas de familias sobre **horarios, costos, inscripción, ubicación, uniforme y equipo '
         'de protección**. También puedes **adjuntar una foto** de un artículo (careta, guantes, peto o '
         'uniforme) y preguntar si sirve, cuánto cuesta o dónde se compra.')

tab_chat, tab_foto, tab_monitor, tab_info = st.tabs(
    ['💬 Asistente', '📷 Identificar artículo', '📈 Monitoreo', 'ℹ️ Uso responsable'])

# ----------------------------------------------------------------------- 1. asistente (agente)
with tab_chat:
    if not st.session_state.mensajes:
        with st.chat_message('assistant'):
            st.write('¡Hola! Soy el asistente de MDK Tuchtlán. Escribe tu pregunta o adjunta una foto '
                     'con el clip 📎.')
    for m in st.session_state.mensajes:
        with st.chat_message(m['rol']):
            if m.get('imagen'):
                st.image(m['imagen'], width=200)
            if m.get('error'):
                st.error(m['texto'])
            elif m['texto']:
                st.write(m['texto'])
            if m.get('detalle'):
                mostrar_detalle(m['detalle'])

    entrada = st.chat_input('Escribe tu pregunta o adjunta una foto (JPG o PNG, máx. 5 MB)',
                            accept_file=True, file_type=['jpg', 'jpeg', 'png'])
    if entrada:
        archivo = entrada.files[0] if entrada.files else None
        if archivo and archivo.size > 5 * 1024 * 1024:
            st.session_state.mensajes.append({'rol': 'assistant', 'error': True,
                                              'texto': 'La foto supera 5 MB. Envía una imagen más ligera.'})
        else:
            enviar_al_agente(entrada.text or '', archivo)
        st.rerun()
    if st.session_state.ejemplo:
        ejemplo, st.session_state.ejemplo = st.session_state.ejemplo, None
        enviar_al_agente(ejemplo)
        st.rerun()

# ----------------------------------------------------------------------- 2. solo clasificador visual
with tab_foto:
    col1, col2 = st.columns(2)
    with col1:
        foto = st.file_uploader('Foto del artículo (JPG o PNG, máx. 5 MB)', type=['jpg', 'jpeg', 'png'])
        if foto is not None:
            if foto.size > 5 * 1024 * 1024:
                st.error('La foto supera 5 MB. Carga una imagen más ligera.')
            else:
                st.image(foto, caption='Foto cargada', width='stretch')
                if st.button('Identificar artículo', type='primary'):
                    with st.spinner('Analizando la imagen con EfficientNetB0…'):
                        r, error = llamar_api('/predict-image',
                                              files={'file': (foto.name, foto.getvalue(), foto.type)})
                    if error:
                        st.error(error)
                    else:
                        st.session_state.prediccion = r
    with col2:
        p = st.session_state.prediccion
        if p is None:
            st.info('Carga una foto y presiona **Identificar artículo**.')
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric('Artículo', NOMBRE_CLASE[p['clase']])
            c2.metric('Confianza', f"{p['confianza']:.0%}")
            c3.metric('Tiempo del modelo', f"{p['latencia_modelo_s'] * 1000:.0f} ms")
            st.progress(min(max(p['confianza'], 0.0), 1.0))
            if p['nivel_confianza'] == 'alta':
                st.success('Confianza alta: el modelo distingue bien el artículo.')
            elif p['nivel_confianza'] == 'media':
                st.warning('Confianza media: revisa las otras opciones antes de decidir.')
            else:
                st.error('Confianza baja: toma otra foto con buena luz, fondo liso y solo el artículo.')
            st.bar_chart(pd.Series(p['probabilidades']).rename(index=NOMBRE_CLASE), height=220)
            st.caption('La foto identifica el TIPO de artículo. No permite comprobar talla, autenticidad, '
                       'estado ni homologación.')

# ----------------------------------------------------------------------- 3. monitoreo
with tab_monitor:
    st.button('Actualizar métricas')
    try:
        m = requests.get(f'{API_URL}/metrics', timeout=5).json()
    except (requests.exceptions.RequestException, ValueError):
        m = None
    if m is None:
        st.error('No se pudieron obtener las métricas: la API no responde.')
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Solicitudes', m['solicitudes_totales'])
        c2.metric('Tasa de error', f"{m['tasa_error']:.1%}")
        c3.metric('Latencia p95', f"{m['latencia_ms']['p95']:.0f} ms")
        conf = m['confianza_visual_promedio']
        c4.metric('Confianza visual promedio', '—' if conf is None else f'{conf:.0%}')
        a, b = st.columns(2)
        with a:
            st.subheader('Decisiones del agente')
            if m['decisiones_agente']:
                st.bar_chart(pd.Series(m['decisiones_agente']).sort_index(), height=220)
        with b:
            st.subheader('Códigos HTTP')
            st.dataframe(pd.DataFrame(m['codigos_http'].items(), columns=['Código', 'Solicitudes']),
                         hide_index=True, width='stretch')
        st.caption(f"Imágenes con confianza baja: {m['imagenes_con_confianza_baja']} · "
                   f"Tiempo activo: {m['tiempo_activo_min']} min · Arranque de la API: {arranque}")

# ----------------------------------------------------------------------- 4. uso responsable
with tab_info:
    st.warning('No subas fotos donde aparezcan rostros de niñas, niños u otras personas: '
               'fotografía solo el artículo.')
    st.markdown("""
**Cómo se tratan tus datos**
- Las fotos se analizan **en memoria** y no se guardan en el servidor. El registro técnico guarda solo
  la hora, el resultado y la confianza; nunca la imagen ni tu mensaje completo.
- La conversación vive solo en **tu sesión** y se borra con **Nueva conversación** o al cerrar la página.

**Qué puede y qué no puede hacer el asistente**
- Reconoce el **tipo** de artículo (careta, guantes, peto, uniforme) y explica precios y reglas de compra.
- **No** comprueba talla, autenticidad, estado ni homologación; eso lo confirma la escuela.
- Es un sistema automatizado y puede equivocarse. Si tienes dudas, escribe a la escuela por WhatsApp.

**Cómo decide (agente híbrido)**

| Situación | Acción |
|---|---|
| Pregunta reconocida por las reglas | Responde con información verificada |
| Las reglas no entienden | El clasificador de intenciones (SVM) propone el tema, o se deriva a una persona |
| Hablas de «este» artículo sin foto | Te pide la foto y recuerda tu pregunta |
| Foto clara | Identifica el artículo y responde tu pregunta |
| Foto dudosa | Pide otra foto |
| Foto y texto no coinciden | Pide aclarar |
| Pregunta de seguimiento | Usa el artículo que se mencionó antes |
""")
