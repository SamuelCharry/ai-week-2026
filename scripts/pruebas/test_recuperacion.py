import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path

import faiss
import numpy as np

from scripts.indice.recuperacion import (
    BM25, Recuperador, crear_ventanas, hash_json, cargar_indice, construir_indice,
    metadatos_busqueda, VERSION_TEXTO_BUSQUEDA,
)
from scripts.indice.sondas import crear_sondas, evaluar_sondas


class Tokenizador:
    def __call__(self, texto, **opciones):
        return {"input_ids": list(range(len(texto))),
                "offset_mapping": [(i, i + 1) for i in range(len(texto))]}


class EncoderMecanico:
    limite = 512
    prefijos = {"passage": "", "query": ""}
    tokenizer = Tokenizador()
    device = "cpu"
    parametros = 0
    precision_real = ["float32"]

    def encode(self, textos, tarea="passage", batch_size=4):
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

    def test_metadatos_sin_seccion_conservan_formato(self):
        esperado = "Prueba mecánica\nArtículo 24\n"
        self.assertEqual(metadatos_busqueda(self.unidad), esperado)
        for seccion in (None, ""):
            with self.subTest(seccion=seccion):
                self.assertEqual(metadatos_busqueda({**self.unidad, "seccion": seccion}), esperado)

    def test_seccion_llega_a_encoder_y_bm25_sin_cambiar_evidencia(self):
        unidad = {**self.unidad, "seccion": "CAPÍTULO II. Competencia territorial"}
        encoder = EncoderMecanico()
        anteriores = crear_ventanas([self.unidad], encoder, 120, 20)
        with tempfile.TemporaryDirectory() as temporal:
            config = {"salida": temporal, "corpus": "corpus_prueba", "tamano_tokens": 120,
                      "solapamiento_tokens": 20, "hibrido": True}
            with patch("scripts.indice.recuperacion.cargar_corpus", return_value=([unidad], {})), \
                    patch.object(encoder, "encode", wraps=encoder.encode) as encode:
                manifiesto, recuperador = construir_indice(config, encoder=encoder)
            self.assertEqual(manifiesto["version_texto_busqueda"], VERSION_TEXTO_BUSQUEDA)
            textos = encode.call_args.args[0]
            self.assertTrue(all("CAPÍTULO II. Competencia territorial\nArtículo 24\n" in t for t in textos))
            self.assertTrue(np.all(recuperador.bm25.buscar("territorial") > 0))
            self.assertEqual(len(anteriores), len(recuperador.ventanas))
            for anterior, ventana in zip(anteriores, recuperador.ventanas):
                for campo in ("texto", "inicio", "fin", "unidad_inicio", "unidad_fin", "fragmento_id"):
                    self.assertEqual(ventana[campo], anterior[campo])
            evidencia = recuperador.buscar("territorial")[0]
            self.assertEqual(evidencia["texto"], self.unidad["texto"])
            self.assertEqual((evidencia["inicio"], evidencia["fin"]), (10, 510))

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
            "from scripts.indice.recuperacion import BM25\n"
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

    def test_indice_sin_version_actual_se_rechaza_antes_de_cargar_modelo(self):
        for version in (None, VERSION_TEXTO_BUSQUEDA - 1, VERSION_TEXTO_BUSQUEDA + 1):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporal:
                manifiesto = {"configuracion": {}, "sha256_archivos": {},
                              "sha256_configuracion": hash_json({})}
                if version is not None:
                    manifiesto["version_texto_busqueda"] = version
                Path(temporal, "manifest.json").write_text(json.dumps(manifiesto), encoding="utf-8")
                with patch("scripts.indice.recuperacion.Encoder") as encoder:
                    with self.assertRaisesRegex(ValueError, "Reconstruir el índice"):
                        cargar_indice(temporal)
                    encoder.assert_not_called()

    def test_experimento_omite_indices_con_texto_busqueda_anterior(self):
        from scripts.experimentos.orquestador import _buscar_indice, _compatible

        ficha = {"repo_id": "prueba", "revision": "revision_prueba"}
        config = {"tamano_tokens": 120, "solapamiento_tokens": 20, "precision": "float32"}
        base = {"configuracion": {**config, "encoder": ficha}, "sha256_corpus": "corpus_prueba"}
        actual = {**base, "version_texto_busqueda": VERSION_TEXTO_BUSQUEDA}
        self.assertTrue(_compatible(actual, ficha, "corpus_prueba", config))
        for version in (None, VERSION_TEXTO_BUSQUEDA - 1, VERSION_TEXTO_BUSQUEDA + 1):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporal:
                manifiesto = dict(base)
                if version is not None:
                    manifiesto["version_texto_busqueda"] = version
                self.assertFalse(_compatible(manifiesto, ficha, "corpus_prueba", config))
                contenido = json.dumps(manifiesto)
                carpeta = Path(temporal)
                (carpeta / "manifest.json").write_text(contenido, encoding="utf-8")
                with zipfile.ZipFile(carpeta / "indice.zip", "w") as archivo:
                    archivo.writestr("indice/manifest.json", contenido)
                with patch("scripts.experimentos.orquestador._verificar_indice") as verificar:
                    self.assertEqual(_buscar_indice(carpeta, [carpeta], ficha, "corpus_prueba", config),
                                     (None, None))
                    verificar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
