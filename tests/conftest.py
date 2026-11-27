import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdk.agente import AgenteMDK                      # noqa: E402
from mdk.nlp import ClasificadorIntenciones           # noqa: E402
from mdk.vision import ClasificadorVisual             # noqa: E402
from evaluacion.simulacion import ModeloFalso         # noqa: E402

FECHA = date(2026, 11, 27)


@pytest.fixture(scope='session')
def nlp():
    return ClasificadorIntenciones()


@pytest.fixture(scope='session')
def agente(nlp):
    return AgenteMDK(nlp, ClasificadorVisual(modelo=ModeloFalso()), fecha=FECHA)


@pytest.fixture(scope='session')
def agente_sin_vision(nlp):
    return AgenteMDK(nlp, ClasificadorVisual(ruta='no_existe.keras'), fecha=FECHA)
