"""Recuperación densa, léxica (BM25) y fusión RRF por unidad citable."""
import math
import re
from collections import Counter, defaultdict

import numpy as np


class BM25:
    def __init__(self, textos):
        self.terminos = [Counter(re.findall(r"\w+", t.casefold())) for t in textos]
        self.longitudes = np.array([sum(t.values()) for t in self.terminos])
        self.promedio = max(1, self.longitudes.mean())
        df = Counter(t for doc in self.terminos for t in doc)
        self.idf = {t: math.log(1 + (len(textos) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.postings = defaultdict(list)
        for i, doc in enumerate(self.terminos):
            for termino, cantidad in doc.items():
                self.postings[termino].append((i, cantidad))

    def buscar(self, consulta):
        scores = np.zeros(len(self.terminos))
        for termino in sorted(set(re.findall(r"\w+", consulta.casefold()))):
            for i, frecuencia in self.postings.get(termino, []):
                norma = 1.2 * (0.25 + 0.75 * self.longitudes[i] / self.promedio)
                scores[i] += self.idf[termino] * frecuencia * 2.2 / (frecuencia + norma)
        return scores


class Recuperador:
    def __init__(self, indice, ventanas, unidades, encoder, hibrido=False):
        self.indice = indice
        self.ventanas = ventanas
        self.unidades = {u["unidad_id"]: u for u in unidades}
        self.encoder = encoder
        self.bm25 = BM25([v["texto_busqueda"] for v in ventanas]) if hibrido else None

    def buscar(self, consulta, k=10):
        vector = self.encoder.encode([consulta], "query")
        scores, posiciones = self.indice.search(vector, len(self.ventanas))
        orden = sorted(zip(posiciones[0].tolist(), scores[0].tolist()), key=lambda x: (-x[1], x[0]))
        if self.bm25:
            lexical = self.bm25.buscar(consulta)
            lexical_orden = sorted(range(len(lexical)), key=lambda i: (-lexical[i], i))
            fusion = defaultdict(float)
            representante = {}
            for ranking in ([i for i, _ in orden], [i for i in lexical_orden if lexical[i] > 0]):
                padres = set()
                for i in ranking:
                    padre = self.ventanas[i]["unidad_id"]
                    if padre in padres:
                        continue
                    padres.add(padre)
                    representante.setdefault(padre, i)
                    fusion[padre] += 1 / (60 + len(padres))
                    if len(padres) == max(100, k):
                        break
            orden = sorted(((representante[padre], valor) for padre, valor in fusion.items()),
                           key=lambda x: (-x[1], x[0]))
        resultado = []
        vistas = set()
        for posicion, score in orden:
            ventana = self.ventanas[posicion]
            if ventana["unidad_id"] in vistas:
                continue
            vistas.add(ventana["unidad_id"])
            unidad = dict(self.unidades[ventana["unidad_id"]])
            unidad.update(puesto=len(resultado) + 1, score=float(score), fragmento_id=ventana["fragmento_id"],
                          ventana_inicio=ventana["inicio"], ventana_fin=ventana["fin"])
            resultado.append(unidad)
            if len(resultado) == k:
                break
        return resultado
