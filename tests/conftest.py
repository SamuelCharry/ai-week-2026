from pathlib import Path

import pytest

from legalrag.corpus import passages_for

FIXTURE = Path(__file__).parent / "fixtures" / "ley_1581_2012.txt"

DOC = {
    "doc_id": "ley_1581_2012",
    "norma": "Ley 1581 de 2012",
    "norma_key": "ley 1581 de 2012",
    "tipo": "Ley estatutaria",
    "numero": "1581",
    "anio": 2012,
    "organo_emisor": "Congreso de la República",
    "vigencia": None,
    "fuente": "Función Pública, Gestor Normativo",
    "url": "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981",
}


@pytest.fixture(scope="session")
def ley_texto():
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def pasajes(ley_texto):
    return passages_for(DOC, ley_texto)
