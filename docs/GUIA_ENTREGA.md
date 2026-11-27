# Guía de entrega del proyecto final (paso a paso)

## 1. Subir el código a GitHub (repositorio `chatbot-taekwondo`)

1. En tu repositorio crea una carpeta `avances/` y mueve ahí lo de los proyectos 1 y 2 (o descárgalo como respaldo).
2. **Add file → Upload files** y arrastra **todo el contenido** de esta carpeta (no la carpeta misma): `app.py`, `api.py`, `requirements.txt`, `Procfile`, `README.md`, `.gitignore`, y las carpetas `mdk/`, `modelos/`, `evaluacion/`, `tests/`, `notebooks/`, `docs/`, `.streamlit/`.
   - El modelo `.keras` pesa unos 17 MB; GitHub acepta archivos de hasta 25 MB por la web.
   - Las carpetas que empiezan con punto (`.streamlit`) a veces no se ven en el explorador de archivos. En Windows activa «Elementos ocultos»; en Mac usa `Cmd + Shift + .`. Si no logras subirla, la app funciona igual (solo cambia el color del tema).
3. *Commit changes*. Revisa que en la página principal del repo se vean `app.py` y la carpeta `mdk/`.

## 2. Ejecutar el notebook en Colab

1. Abre `notebooks/MDK_Final_Colab.ipynb` en Colab (desde GitHub: *File → Open notebook → GitHub*).
2. Ejecuta las celdas en orden. Cuando lo pida, autoriza Google Drive: el notebook busca `MDK_imagenes_clasificadas.zip` en `MyDrive/Chatbot_MDK_Tuchtlan/` (donde lo guardaste en el avance 2).
3. **Muy recomendado:** junta entre 15 y 20 fotos **nuevas** (de otras personas o de otro lugar, que nunca se usaron para entrenar), ordénalas en carpetas `casco/`, `guantes/`, `peto/` y `uniforme/`, comprímelas como `fotos_nuevas.zip` y súbelas a la misma carpeta de Drive. Esa métrica responde a la limitación que tú misma señalaste en el avance 2.
4. Al final descarga `evidencias_mdk.zip`: trae las matrices de confusión, las tablas de métricas, las pruebas de la API y del agente, la prueba de carga, el diagrama de arquitectura y el log.

## 3. Desplegar en Streamlit Community Cloud (URL permanente)

1. Entra a https://share.streamlit.io con tu cuenta de GitHub.
2. **Create app → Deploy a public app from GitHub**. Repositorio `fergordillo880109/chatbot-taekwondo`, rama `main`, archivo `app.py`.
3. **Advanced settings → Python 3.12**. Guarda y presiona **Deploy**.
4. La primera instalación tarda varios minutos por TensorFlow. Toma capturas del proceso y del resultado.
5. Copia la URL (`https://….streamlit.app`) y pégala en el `README.md`.
6. Si la app marca «Servicio parcial» o se reinicia por falta de memoria, revisa *Manage app → Logs*. Mientras tanto, la URL de Cloudflare del notebook sirve para el video, y en el reporte se documenta como hallazgo de despliegue.

## 4. Capturas obligatorias

- [ ] App en la URL pública con «Servicio de IA en línea».
- [ ] **Integración final funcionando** (pedido del profesor): «Tengo este, ¿lo puedo llevar?» → pide la foto → adjuntas la foto → responde → «¿y cuánto cuesta?» → responde con la memoria. Abre «¿Cómo decidió el asistente?».
- [ ] Chatbot respondiendo al menos 3 preguntas (horarios, costos, una resuelta por el SVM, como «¿Hay lugares disponibles para nuevos alumnos?»).
- [ ] Pestaña «Identificar artículo» con probabilidades.
- [ ] Pestaña «Monitoreo».
- [ ] Un error bien comunicado (foto de más de 5 MB o archivo que no es imagen).
- [ ] Panel de Streamlit Cloud y repositorio de GitHub.
- [ ] Salida de `pytest` y tablas del notebook.

## 5. Guion del video (5 a 10 minutos)

| Tiempo | Contenido |
|---|---|
| 0:00 | Presentación, la escuela y el problema de las familias |
| 1:00 | Diagrama de arquitectura: Streamlit → Flask → agente → reglas/SVM/CNN |
| 2:00 | Demo en la URL pública: horarios (R1), «¿Hay lugares disponibles para nuevos alumnos?» (R2), «asdf» (R3) |
| 3:30 | «Tengo este, ¿lo puedo llevar?» (R4) → foto (R5) → «¿y qué tallas hay?» (R8 + límite) |
| 5:00 | Foto confusa (R6) y contradicción «este casco» con foto de peto (R7) |
| 6:00 | Pestaña Identificar artículo, Monitoreo y un error de archivo |
| 7:00 | GitHub, Streamlit Cloud y pruebas automáticas |
| 8:00 | Métricas, ética, limitaciones y mejoras futuras |

## 6. De dónde sale cada sección del documento (15 a 20 cuartillas)

| Sección | Fuente |
|---|---|
| Introducción y problema | Avance 2 §1–2 (actualizado) |
| Arquitectura general | `resultados/arquitectura.png` + README |
| Módulo de NLP | Avance 2 §4–5 + `intenciones_resumen.csv` (82 → 202 frases; prueba independiente) |
| Módulo de visión | Avance 2 §6–7 + `vision_prueba_*` + `vision_fotos_nuevas_*` |
| Agente inteligente | Tabla R0–R9 de `mdk/agente.py` + `pruebas_agente.csv` |
| Desarrollo web | Capturas de la app + `app.py` (session_state, caché, manejo de errores) |
| API REST | Tabla de endpoints del README + `pruebas_api.csv` |
| Despliegue | Capturas de GitHub, Streamlit Cloud y el túnel |
| Resultados | Todas las tablas + `prueba_carga.png` + `metrics.json` |
| Ética | Pestaña «Uso responsable» + menores de edad, sobreconfianza y precios |
| Conclusiones y mejoras | Ver la lista de abajo |

**Corrige en el texto:** el avance 2 decía *SnowballStemmer*, pero el código usa la función propia `stem_ligero`.

**Mejoras futuras sugeridas:** fotos reales de la escuela y de más personas; detección con YOLO para varias piezas en una foto; un modelo generativo que redacte con base en la información verificada; un panel para que la escuela actualice precios; límite de solicitudes por usuario (429); convertir la CNN a TFLite para reducir memoria; recolectar preguntas reales de familias (con consentimiento) para reentrenar.
