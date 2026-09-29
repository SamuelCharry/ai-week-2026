"""src/main.py: búsqueda y enlace de datos, sin GPU."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legalrag.config import RAIZ, leer_config

spec = importlib.util.spec_from_file_location("main_entrega", RAIZ / "src/main.py")
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


class MainTest(unittest.TestCase):
    def test_encuentra_y_enlaza_datos_anidados(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            repo, datos = tmp / "repo", tmp / "datos/ai-week-2026"
            (datos / "data/processed/corpus_preparado/textos").mkdir(parents=True)
            (datos / "data/processed/corpus_preparado/textos/ley_1_2000.txt").write_text("x")
            (datos / "data/releases/corpus_eval_v1").mkdir(parents=True)
            (datos / "data/releases/corpus_eval_v1/corpus_manifest.json").write_text("[]")
            (datos / "reports/reporte_evaluacion").mkdir(parents=True)
            (datos / "reports/reporte_evaluacion/modelos_verificados.json").write_text("[]")
            (datos / "e06-resultados/indices").mkdir(parents=True)
            (datos / "e06-resultados/chunks.sqlite").write_text("")
            repo.mkdir()
            with patch.object(main, "RAIZ", repo):
                main.preparar_datos(tmp / "datos", leer_config())
                self.assertTrue((repo / "data/processed/corpus_preparado/textos/ley_1_2000.txt").is_file())
                self.assertTrue((repo / "data/releases/corpus_eval_v1/corpus_manifest.json").is_file())
                self.assertTrue((repo / "reports/reporte_evaluacion/modelos_verificados.json").is_file())
                self.assertTrue((repo / "data/experimentos/corpus_definitivo/chunks.sqlite").is_file())
                self.assertEqual(main.enlazar(datos / "data/releases/corpus_eval_v1",
                                              "data/releases/corpus_eval_v1"), "ya estaba")

    def test_sin_textos_explica_que_falta(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(main, "RAIZ", Path(tmp) / "repo"), self.assertRaises(SystemExit):
                main.preparar_datos(Path(tmp), leer_config())


if __name__ == "__main__":
    unittest.main()
