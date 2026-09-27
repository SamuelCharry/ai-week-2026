import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import faiss
import numpy as np

from scripts.auxiliares.recuperacion import BM25, Recuperador, crear_ventanas, hash_json, cargar_indice, construir_indice
from scripts.auxiliares.sondas import crear_sondas, evaluar_sondas


class Tokenizador:
    def __call__(self, texto, **opciones):
        return {"input_ids": list(range(len(texto))),
                "offset_mapping": [(i, i + 1) for i in range(len(texto))]}


class EncoderMecanico:
    limite = 512
    prefijos = {"passage": "", "query": ""}
    tokenizer = Tokenizador()

    def encode(self, textos, tarea="passage"):
        return np.array([[1, 0] for _ in textos], dtype="float32")


class RecuperacionTest(unittest.TestCase):
    def setUp(self):
        self.unidad = {"unidad_id": "prueba_1", "doc_id": "prueba", "articulo": "24",
                       "inicio": 10, "fin": 510, "texto": "abcde" * 100, "titulo": "Prueba mecánica",
                       "areas": ["prueba"]}

    def test_ventanas_conservan_texto_y_cobertura(self):
        ventanas = crear_ventanas([self.unidad], EncoderMecanico(), 120, 20)
        self.assertEqual(ventanas[0]["inicio"], 10)
        self.assertEqual(ventanas[-1]["fin"], 510)
        for ventana in ventanas:
            self.assertEqual(ventana["texto"], self.unidad["texto"][ventana["inicio"] - 10:ventana["fin"] - 10])
            self.assertEqual(ventana["unidad_id"], self.unidad["unidad_id"])
        for anterior, siguiente in zip(ventanas, ventanas[1:]):
            self.assertLessEqual(siguiente["inicio"], anterior["fin"])

    def test_solapamiento_invalido(self):
        with self.assertRaises(ValueError):
            crear_ventanas([self.unidad], EncoderMecanico(), 100, 100)

    def test_recupera_padre_sin_repetir(self):
        ventanas = crear_ventanas([self.unidad], EncoderMecanico(), 120, 20)
        indice = faiss.IndexFlatIP(2)
        indice.add(np.array([[1, 0]] * len(ventanas), dtype="float32"))
        r = Recuperador(indice, ventanas, [self.unidad], EncoderMecanico())
        pasajes = r.buscar("abcde")
        self.assertEqual(len(pasajes), 1)
        self.assertEqual(pasajes[0]["texto"], self.unidad["texto"])
        self.assertEqual(pasajes[0]["inicio"], 10)
        self.assertEqual(pasajes[0]["fin"], 510)
        sondas = crear_sondas([self.unidad])
        resultado = evaluar_sondas(r, sondas)
        self.assertTrue(all(x["recall_1"] == 1 for x in resultado))

    def test_bm25_distingue_numeros(self):
        scores = BM25(["artículo 24", "artículo 42"]).buscar("artículo 42")
        self.assertGreater(scores[1], scores[0])

    def test_bm25_no_depende_de_semilla_hash(self):
        codigo = (
            "from scripts.auxiliares.recuperacion import BM25\n"
            "textos = [' '.join('termino' + str(j) for j in range(i, i + 30)) for i in range(50)]\n"
            "consulta = ' '.join('termino' + str(j) for j in range(70))\n"
            "print(BM25(textos).buscar(consulta).tobytes().hex())"
        )
        salidas = []
        for semilla in ("0", "13"):
            entorno = {**os.environ, "PYTHONHASHSEED": semilla}
            salida = subprocess.check_output([sys.executable, "-c", codigo], env=entorno)
            salidas.append(salida)
        self.assertEqual(salidas[0], salidas[1])

    def test_hash_independiente_orden_claves(self):
        self.assertEqual(hash_json({"a": 1, "b": 2}), hash_json({"b": 2, "a": 1}))

    def test_hibrido_no_pierde_padres_por_ventanas_repetidas(self):
        unidades = [{**self.unidad, "unidad_id": str(i)} for i in range(10)]
        ventanas = []
        for i, unidad in enumerate(unidades):
            for j in range(200 if i == 0 else 1):
                ventanas.append({**unidad, "texto_busqueda": "abcde", "fragmento_id": f"{i}_{j}"})
        indice = faiss.IndexFlatIP(2)
        indice.add(np.array([[1, 0]] * len(ventanas), dtype="float32"))
        r = Recuperador(indice, ventanas, unidades, EncoderMecanico(), hibrido=True)
        self.assertEqual(len(r.buscar("abcde", 10)), 10)

    def test_sonda_rechaza_offsets_desplazados(self):
        ventanas = crear_ventanas([self.unidad], EncoderMecanico(), 120, 20)
        indice = faiss.IndexFlatIP(2)
        indice.add(np.array([[1, 0]] * len(ventanas), dtype="float32"))
        r = Recuperador(indice, ventanas, [self.unidad], EncoderMecanico())
        corrupto = r.buscar("abcde")[0]
        corrupto.update(inicio=11, fin=511)
        with patch.object(r, "buscar", return_value=[corrupto]):
            with self.assertRaises(ValueError):
                evaluar_sondas(r, crear_sondas([self.unidad]))

    def test_flatip_serializacion_exacta(self):
        indice = faiss.IndexFlatIP(2)
        indice.add(np.eye(2, dtype="float32"))
        with tempfile.TemporaryDirectory() as temporal:
            archivo = str(Path(temporal) / "indice.faiss")
            faiss.write_index(indice, archivo)
            nuevo = faiss.read_index(archivo)
            valores, ids = nuevo.search(np.array([[0, 1]], dtype="float32"), 2)
            self.assertEqual(ids.tolist(), [[1, 0]])
            self.assertEqual(valores.tolist(), [[1, 0]])

    def test_configuracion_modificada_se_rechaza_antes_de_cargar_modelo(self):
        with tempfile.TemporaryDirectory() as temporal:
            manifiesto = {"configuracion": {"hibrido": True}, "sha256_archivos": {},
                          "sha256_configuracion": hash_json({"hibrido": False})}
            Path(temporal, "manifest.json").write_text(json.dumps(manifiesto), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuración"):
                cargar_indice(temporal)

    def test_no_sobrescribe_indice_congelado(self):
        with tempfile.TemporaryDirectory() as temporal:
            Path(temporal, "CONGELADO.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "congelado"):
                construir_indice({"salida": temporal})


if __name__ == "__main__":
    unittest.main()
