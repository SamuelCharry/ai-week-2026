import json
import tempfile
import unittest
from pathlib import Path

import nbformat

from legalrag.experimentos import corpus_definitivo as exp


class NotebookDefinitivoTest(unittest.TestCase):
    def test_notebook_y_entrada_sin_clave(self):
        notebook = nbformat.read(exp.ROOT / "notebooks/experimentos/e06_corpus_definitivo.ipynb", as_version=4)
        nbformat.validate(notebook)
        self.assertGreaterEqual(len(notebook.cells), 12)
        code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        self.assertNotIn("dummy_code", code)
        self.assertIn("preflight['snapshot_id']", code)
        self.assertIn("score['cerradas']['puntos'] <= 20", code)
        items = exp.questions()
        self.assertEqual(len(items), 50)
        for q in items:
            self.assertNotIn("legal_basis", q)
            self.assertNotIn("respuesta_correcta", q)
            self.assertNotIn("respuesta_esperada", q)

    def test_piloto_estructura_literal_y_bm25(self):
        old = exp.RUNS
        with tempfile.TemporaryDirectory() as temp:
            exp.RUNS = Path(temp)
            try:
                meta = exp.build_lexical(limit_docs=2)
                self.assertEqual(meta["documentos"], 2)
                self.assertGreater(meta["chunks"], 0)
                doc = json.loads((exp.RELEASE / "corpus_manifest.json").read_text(encoding="utf-8"))[0]
                q = {"pregunta": doc["titulo"], "opciones": {}}
                rank = exp.bm25(q, 10)
                self.assertTrue(rank)
                passage = exp.passages(rank[:1])[0]
                text = (exp.ROOT / doc["texto_archivo"]).read_text(encoding="utf-8")
                if passage["doc_id"] == doc["doc_id"]:
                    self.assertEqual(passage["texto"], text[passage["inicio"]:passage["fin"]])
                self.assertEqual(meta, exp.build_lexical(limit_docs=2))
            finally:
                exp.RUNS = old


if __name__ == "__main__":
    unittest.main()
