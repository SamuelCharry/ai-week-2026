"""Ampliación del índice con fuentes puntuales: ids consecutivos, FTS y FAISS alineados, sin duplicar."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import faiss
    import numpy as np
except ImportError:  # sin faiss no hay índice que ampliar
    faiss = None

from legalrag.ingestion import agregar_puntuales as ap

TEXTO = ("DECRETO 1572 DE 2024\n(diciembre 24)\nPor el cual se fija el salario mínimo mensual legal.\n"
         "ARTÍCULO 1. Fijar a partir del primero (1) de enero de 2025 como Salario Mínimo Legal Mensual "
         "la suma de un millón cuatrocientos veintitrés mil quinientos pesos ($1.423.500).\n"
         "ARTÍCULO 2. El presente decreto rige a partir del primero (1) de enero de 2025.\n")


def indice_minimo(raiz, dimension=4):
    carpeta = raiz / "run"
    (carpeta / "idx").mkdir(parents=True)
    conexion = sqlite3.connect(carpeta / "chunks.sqlite")
    conexion.executescript("""
    CREATE TABLE chunks (
      id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, titulo TEXT NOT NULL,
      tipo TEXT NOT NULL, articulo TEXT, seccion TEXT, unidad_id TEXT NOT NULL,
      unidad_inicio INTEGER NOT NULL, unidad_fin INTEGER NOT NULL,
      inicio INTEGER NOT NULL, fin INTEGER NOT NULL, texto TEXT NOT NULL,
      texto_busqueda TEXT NOT NULL, avisos TEXT NOT NULL);
    CREATE VIRTUAL TABLE fts USING fts5(texto_busqueda, content='chunks', content_rowid='id',
      tokenize='unicode61 remove_diacritics 2');
    """)
    for i in (1, 2):
        conexion.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (i, f"ley_{i}_2000", "Ley", "ley", None, None, "u", 0, 1, 0, 1, "texto", "texto", "[]"))
        conexion.execute("INSERT INTO fts(rowid,texto_busqueda) VALUES(?,?)", (i, "texto"))
    conexion.commit()
    conexion.close()
    indice = faiss.IndexFlatIP(dimension)
    indice.add(np.ones((2, dimension), dtype="float32"))
    faiss.write_index(indice, str(carpeta / "idx/index.faiss"))
    (carpeta / "idx/complete.json").write_text(json.dumps({"chunks": 2}), encoding="utf-8")
    (carpeta / "chunks_complete.json").write_text(json.dumps({"chunks": 2}), encoding="utf-8")
    (raiz / "textos").mkdir()
    (raiz / "textos/decreto_1572_2024.txt").write_text(TEXTO, encoding="utf-8")
    rec = {"fragmentos": "run/chunks.sqlite", "indice_denso": "run/idx/index.faiss", "indice_meta": "run/idx/complete.json"}
    fila = {"doc_id": "decreto_1572_2024", "titulo": "Salario mínimo mensual legal para el año 2025", "tipo": "decreto",
            "numero": "1572", "anio": 2024, "texto_archivo": "textos/decreto_1572_2024.txt"}
    return rec, fila


@unittest.skipIf(faiss is None, "requiere faiss")
class AmpliarIndice(unittest.TestCase):
    def setUp(self):
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        self.raiz = Path(carpeta.name)
        parche = mock.patch.object(ap, "RAIZ", self.raiz)
        parche.start()
        self.addCleanup(parche.stop)

    def test_agrega_fragmentos_y_vectores_alineados(self):
        tmp_path = self.raiz
        rec, fila = indice_minimo(tmp_path)
        codificados = []

        def codificar(texto):
            codificados.append(texto)
            return np.full((1, 4), len(codificados), dtype="float32")

        por_doc = ap.agregar_al_indice([fila], rec, codificar)
        nuevos = por_doc["decreto_1572_2024"]
        assert nuevos >= 1 and len(codificados) == nuevos

        conexion = sqlite3.connect(tmp_path / "run/chunks.sqlite")
        ids = [r[0] for r in conexion.execute("SELECT id FROM chunks WHERE doc_id = 'decreto_1572_2024' ORDER BY id")]
        assert ids == list(range(3, 3 + nuevos))
        # BM25 encuentra el valor del salario en los fragmentos nuevos.
        assert conexion.execute("SELECT COUNT(*) FROM fts WHERE fts MATCH '\"salario\"'").fetchone()[0] >= 1
        indice = faiss.read_index(str(tmp_path / "run/idx/index.faiss"))
        assert indice.ntotal == 2 + nuevos
        # Posición i del índice = fragmento i + 1: el primer vector nuevo es el del fragmento 3.
        assert indice.reconstruct(2)[0] == 1.0
        assert json.loads((tmp_path / "run/idx/complete.json").read_text())["chunks"] == 2 + nuevos

        # Correrlo otra vez no duplica nada.
        assert ap.agregar_al_indice([fila], rec, codificar) == {}
        assert faiss.read_index(str(tmp_path / "run/idx/index.faiss")).ntotal == 2 + nuevos

    def test_no_toca_un_indice_desalineado(self):
        tmp_path = self.raiz
        rec, fila = indice_minimo(tmp_path)
        indice = faiss.read_index(str(tmp_path / rec["indice_denso"]))
        indice.add(np.ones((1, 4), dtype="float32"))
        faiss.write_index(indice, str(tmp_path / rec["indice_denso"]))
        with self.assertRaisesRegex(RuntimeError, "inconsistente"):
            ap.agregar_al_indice([fila], rec, lambda texto: np.ones((1, 4), dtype="float32"))
        assert sqlite3.connect(tmp_path / "run/chunks.sqlite").execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2


if __name__ == "__main__":
    unittest.main()
