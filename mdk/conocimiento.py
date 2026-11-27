"""Base de conocimiento de MDK Tuchtlán.

Es el ÚNICO lugar donde se actualizan precios, horarios, dirección y promociones.
Datos confirmados con la responsable de la escuela (2026). Los precios deben
verificarse antes de difundir la aplicación al público.
"""

# MÓDULO 2 — Base de conocimiento editable
ESCUELA = {
    'nombre': 'MDK Tuchtlán',
    'direccion': 'Boulevard Paso Limón #1621, colonia Paso Limón, Tuxtla Gutiérrez, Chiapas, a un costado del OXXO',
    'whatsapp': '961-654-4587',
    'enlace_whatsapp': 'https://wa.me/529616544587',
    'costos': {'inscripcion': 300, 'mensualidad': 600, 'uniforme': 1000,
               'examen_grado': 750, 'cinta_kup': 135, 'cuello_kup': 90,
               'reforzamiento_preescolar': 300},
    'equipo': {'peto_00_4': 905, 'peto_5': 1015, 'careta': 850,
               'codera': 605, 'espinillera': 605, 'concha': 525,
               'guantes': 585, 'empeineras': 585, 'bucal': 200,
               'mochila': 930, 'paquete': 3490, 'paquete_mochila': 4420},
    # Edad excepcional de “2.8 meses” pendiente de aclarar. No automatizar como admisión confirmada.
    'edad_regular': 3,
    'horarios': '¡Claro! Tenemos clases de lunes a jueves: 5:00–6:00 p. m., Preescolar e Infantil Menor (aprox. 3–9 años); 6:00–7:00 p. m., Infantil Mayor y Juveniles (aprox. 9 años en adelante). Te ayudamos a elegir el grupo según la edad y el nivel del alumno; a los 9 años valoramos cuál de los dos turnos le conviene.',
    'promocion_octubre': {'inicio': '2026-10-01', 'fin': '2026-10-31', 'descuento_uniforme': 0.20},
    'promocion_nov_dic': {'inicio': '2026-11-01', 'fin': '2026-12-31',
                         'descuento_uniforme': 0.25, 'inscripcion_gratis': False},
}
CONTACTO = f"Escríbenos por WhatsApp al {ESCUELA['whatsapp']} y con gusto te atenderemos personalmente."
