"""Genera CORPUS.md y corpus_manifest.json a partir del corpus procesado.

Sigue la plantilla oficial de data/oficial/entregables/sabado/CORPUS.md: inventario
por documento, criterio frente a la composición del banco, método de ingesta,
evolución del puntaje y licencia. Todo lo derivable se deriva del corpus y de
seed_targets.json; la tabla de evolución se toma de configs/evolucion.json, que se
actualiza con cada corrida evaluada.

Se dejaban a mano y quedaban desactualizados. Con esto son un producto del corpus.

Uso: python -m scripts.corpus.manifiesto [--equipo NOMBRE] [--enlace URL]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ))

CORPUS = RAIZ / "data/processed/corpus"
LICENCIA = "CC-BY-4.0"

# Composición del banco de 1.042 ítems, sección 4.2 del enunciado. No hay una
# versión legible por máquina en el material oficial (falta questions_export.json).
ITEMS_BANCO = {
    "Derecho constitucional": 134, "Derecho administrativo": 124, "Derecho penal": 123,
    "Derecho procesal": 111, "Derecho comercial y sociedades": 104, "Derecho civil": 102,
    "Derecho de familia": 93, "Derecho tributario": 92, "Derecho laboral": 87,
    "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]": 72,
}
NOMBRE_CORTO = {
    "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]":
        "Derecho de los mercados",
}
# El corpus escribe "comercial_sociedades" donde el material oficial usa "comercial".
ALIAS_AREA = {"comercial_sociedades": "comercial"}

# El nombre de la Función Pública entró con cuatro grafías (tildes, guion largo,
# abreviado). Se unifican al escribir el manifiesto, sin tocar los originales.
def fuente_canonica(nombre):
    nombre = (nombre or "").strip()
    sin_tildes = nombre.translate(str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU"))
    if "Gestor Normativo" in nombre or "Funcion Publica" in sin_tildes:
        return "Departamento Administrativo de la Función Pública - Gestor Normativo"
    return nombre.replace("—", "-")


def leer_jsonl(ruta, campos=None):
    with Path(ruta).open(encoding="utf-8") as archivo:
        for linea in archivo:
            if linea.strip():
                fila = json.loads(linea)
                yield {k: fila.get(k) for k in campos} if campos else fila


def area_oficial(slug):
    """Slug del corpus -> nombre de área del material oficial."""
    from common import AREA_SLUG

    slug = ALIAS_AREA.get(slug, slug)
    return next((nombre for nombre, s in AREA_SLUG.items() if s == slug), None)


def metodo(documento):
    """Cómo se obtuvo el texto de este documento, en una frase."""
    formatos = {Path(a["archivo"]).suffix.lower() for a in documento.get("archivos_raw") or []}
    partes = []
    if {".html", ".htm"} & formatos:
        partes.append("parser HTML")
    if ".pdf" in formatos:
        partes.append("extracción de PDF")
    corte = "segmentación por artículo" if documento.get("tipo") != "sentencia" else "segmentación por bloque"
    tramos = len(documento.get("archivos_raw") or [])
    detalle = f" sobre {tramos} tramos" if tramos > 1 else ""
    return f"{' + '.join(partes) or 'texto plano'}{detalle} + {corte}"


def reunir():
    """Documentos con sus conteos, más los totales del corpus."""
    documentos = list(leer_jsonl(CORPUS / "documentos.jsonl"))
    articulos, fragmentos = defaultdict(set), Counter()
    for unidad in leer_jsonl(CORPUS / "unidades.jsonl", ["doc_id", "tipo_unidad", "articulo"]):
        if unidad["tipo_unidad"] == "articulo" and unidad["articulo"]:
            articulos[unidad["doc_id"]].add(str(unidad["articulo"]))
    for fragmento in leer_jsonl(CORPUS / "fragmentos.jsonl", ["doc_id"]):
        fragmentos[fragmento["doc_id"]] += 1
    resumen = json.loads((CORPUS / "resumen.json").read_text(encoding="utf-8"))
    return documentos, articulos, fragmentos, resumen


def fichas(documentos, articulos, fragmentos):
    salida = []
    for d in sorted(documentos, key=lambda x: x["doc_id"]):
        areas = [a for a in (area_oficial(s) for s in d.get("areas") or []) if a]
        salida.append({
            "doc_id": d["doc_id"], "titulo": d.get("titulo"), "fuente": fuente_canonica(d.get("fuente")),
            "url": d.get("url"), "fecha_consulta": d.get("fecha_consulta"), "areas": areas,
            "n_articulos": len(articulos[d["doc_id"]]) or None,
            "n_fragmentos": fragmentos[d["doc_id"]],
            "metodo_ingesta": metodo(d), "sha256": d.get("sha256_texto"),
        })
    return salida


def cobertura_por_area():
    """Menciones del banco cubiertas por área, según seed_targets y el corpus."""
    from scripts.corpus.cobertura import auditar_banco, indice_corpus

    total, cubierto, normas, normas_ok = Counter(), Counter(), Counter(), Counter()
    for fila in auditar_banco(RAIZ, indice=indice_corpus(RAIZ)):
        for area in fila["areas"]:
            total[area] += fila["items_del_banco"]
            normas[area] += 1
            if fila["en_corpus"]:
                cubierto[area] += fila["items_del_banco"]
                normas_ok[area] += 1
    return total, cubierto, normas, normas_ok


def tabla_inventario(fichas_docs):
    lineas = ["| doc_id | Título | Fuente | URL | Fecha de consulta | Artículos | Áreas |",
              "|---|---|---|---|---|---:|---|"]
    for f in fichas_docs:
        areas = ", ".join(NOMBRE_CORTO.get(a, a).replace("Derecho ", "").capitalize() for a in f["areas"])
        titulo = (f["titulo"] or "").replace("|", "/")
        lineas.append(f"| `{f['doc_id']}` | {titulo} | {f['fuente'] or ''} | `{f['url'] or ''}` | "
                      f"{f['fecha_consulta'] or ''} | {f['n_articulos'] or ''} | {areas} |")
    return "\n".join(lineas)


def tabla_criterio(total, cubierto, normas, normas_ok):
    lineas = ["| Área | Ítems en el banco | Normas objetivo en el corpus | Menciones cubiertas | Cobertura |",
              "|---|---:|---:|---:|---:|"]
    for area, items in sorted(ITEMS_BANCO.items(), key=lambda kv: -kv[1]):
        t, c = total.get(area, 0), cubierto.get(area, 0)
        porcentaje = f"{100 * c / t:.0f} %" if t else "—"
        lineas.append(f"| {NOMBRE_CORTO.get(area, area)} | {items} | "
                      f"{normas_ok.get(area, 0)}/{normas.get(area, 0)} | {c}/{t} | {porcentaje} |")
    return "\n".join(lineas)


def tabla_evolucion():
    ruta = RAIZ / "configs/evolucion.json"
    if not ruta.is_file():
        return "Pendiente de registrar en `configs/evolucion.json`."
    filas = json.loads(ruta.read_text(encoding="utf-8"))
    lineas = ["| Fecha | Experimento | Variante | Documentos | Fragmentos | Cerradas /20 | "
              "Citación /20 | Abstención /10 | Total /50 | Qué cambió |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for f in filas:
        def v(clave):
            valor = f.get(clave)
            return "—" if valor is None else f"{valor:.1f}".replace(".", ",")
        lineas.append(f"| {f.get('fecha', '')} | {f.get('experimento', '')} | {f.get('variante', '')} | "
                      f"{f.get('documentos', '')} | {f.get('fragmentos', '')} | "
                      f"{v('cerradas')} | {v('citas')} | {v('abstencion')} | {v('total')} | {f.get('cambio', '')} |")
    return "\n".join(lineas)


def escribir_corpus_md(fichas_docs, resumen, cobertura, encoder):
    total, cubierto, normas, normas_ok = cobertura
    fuentes = Counter(f["fuente"] or "sin declarar" for f in fichas_docs)
    tipos = Counter()
    for f in fichas_docs:
        tipos["con artículos" if f["n_articulos"] else "sin artículos"] += 1
    texto = f"""# Bitácora del corpus

Corpus de derecho colombiano construido para el reto. Un registro por documento,
con la URL y la fecha con que se descargó, de modo que se pueda reconstruir.

## 1. Inventario

{tabla_inventario(fichas_docs)}

**Totales**

| Métrica | Valor |
|---|---:|
| Documentos incorporados | {len(fichas_docs)} |
| Artículos con unidad propia | {resumen['unidades_por_tipo'].get('articulo', 0)} |
| Unidades legales | {resumen['unidades']} |
| Fragmentos en el índice | {resumen['fragmentos']} |
| Documentos con artículos numerados | {tipos['con artículos']} |
| Jurisprudencia y actos sin artículos | {tipos['sin artículos']} |

Fuentes: {", ".join(f"{n} de {f}" for f, n in fuentes.most_common())}.

## 2. Criterio de selección

Se parte de `seed_targets.json`, que declara las normas que el banco de 1.042
ítems cita y cuántas veces. Se incorpora primero lo más citado y se completa por
área. La columna de menciones cubiertas pesa cada norma por sus ítems del banco.

{tabla_criterio(total, cubierto, normas, normas_ok)}

Un documento puede pertenecer a varias áreas, así que las filas no suman el total.

Qué señal se usó para decidir qué descargar, y qué no entró al sistema: se leyó el
nombre de las normas de `seed_targets.json` y el campo `legal_basis` de la muestra
pública de 50, que cita nueve cuerpos normativos que ese archivo no lista. Esa lectura
solo sirvió para saber qué texto oficial buscar. Ni las preguntas, ni las opciones, ni
las respuestas esperadas, ni el propio `legal_basis` entran al corpus, al índice o al
contexto del modelo.

Quedaron fuera, con su motivo:

- Normas que la Secretaría del Senado no publica en su base documental. Se
  registran en `data/raw/ampliacion_v05.csv` con el código HTTP recibido.
- Providencias de la Corte Suprema que la relatoría solo publica dentro del
  boletín del mes y no como archivo propio.
- Entradas del banco cuyo identificador no corresponde a una norma existente
  (por ejemplo "Ley 11500 de 2007" o "Ley 116 de 2006"). No se sustituyen por la
  norma que parecen querer decir: el evaluador compara el identificador literal.
- Tres originales con defectos de texto confirmados, que se conservan sin indexar:
  {", ".join(resumen.get('documentos_fuente_restringida') or []) or "ninguno"}.

## 3. Método de ingesta y limpieza

1. **Descarga.** Todos los originales vienen de publicadores oficiales del Estado
   colombiano, sin intermediarios ni recopilaciones privadas. La lista completa, con
   cuántos documentos aporta cada uno, está al final del inventario. La ampliación de
   esta semana usó la relatoría de la Corte Constitucional, la relatoría de la Corte
   Suprema y la base documental de la Secretaría General del Senado; el corpus inicial
   se había armado además con el Gestor Normativo de la Función Pública y con las
   compilaciones de DIAN, SIC, Consejo de Estado y varios ministerios.
   Cada descarga se valida contra el identificador esperado, para no guardar una página
   de error que responde 200. Las normas que Senado parte en varias páginas se siguen
   por la cadena `_prNNN`.
2. **Extracción de texto.** Parser HTML propio y extracción de PDF con pdfplumber.
   Se conserva la marca de página de los PDF.
3. **Normalización.** Reparación de codificación y de entidades HTML; retiro de la
   cabecera editorial repetida, con rastro por página. No se reescribe el texto.
4. **Segmentación.** Las normas se cortan por artículo; la jurisprudencia por
   párrafo, sección o bloque identificado. Cada unidad guarda su offset sobre el
   texto normalizado, y las ventanas del índice localizan la unidad sin partirla.
   Las remisiones editoriales entre paréntesis ("(Ver Ley 388 de 1997; Art. 1.)")
   no se toman como cabeceras de artículo.
5. **Extracción de metadatos.** Tipo, número, año, órgano, áreas, señales de
   vigencia declaradas por la fuente y hash del original.
6. **Indexación.** {encoder.get('nombre')} ({encoder.get('dimensiones')} dimensiones),
   FAISS IndexFlatIP con vectores normalizados.

Problemas encontrados y cómo se resolvieron:

- La relatoría de la Corte Constitucional publica las SU sin guion (`su016-20`) y
  las C y T con guion (`c-207-19`). Pedirlas con el patrón equivocado devuelve la
  portada del sitio con estado 200. Se corrigió el patrón y la validación.
- Varios encabezados llevan anotaciones intercaladas ("DECRETO &lt;LEY&gt; 4334 DE
  2008", "LEY ESTATUTARIA 1581 DE 2012") que ningún extractor de citas reconoce.
  Cuando el documento nunca se nombra a sí mismo de forma citable, la evidencia
  incluye su cita canónica tomada del manifiesto, marcada como tal.
- Las notas "(Ver Ley X; Art. N)" del Gestor Normativo desordenaban la numeración
  y dejaban 37 artículos de la Constitución sin unidad propia. Corregido: los 380
  quedan con unidad.
- La Constitución se volvió a sembrar desde la Secretaría del Senado. El original
  anterior se conserva sin referenciar.

## 4. Evolución del puntaje

{tabla_evolucion()}

## 5. Licencia

El corpus se publica bajo `{LICENCIA}`. Los textos normativos colombianos son de
dominio público; la licencia cubre el procesamiento, la segmentación y la
extracción de metadatos hechos por el equipo. Las notas y ediciones de terceros no
se relicencian.

Las preguntas de evaluación y sus respuestas no entran al corpus ni al índice.
"""
    (RAIZ / "CORPUS.md").write_text(texto, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equipo", default="")
    ap.add_argument("--enlace", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(RAIZ / "data/oficial/scripts"))
    documentos, articulos, fragmentos, resumen = reunir()
    fichas_docs = fichas(documentos, articulos, fragmentos)
    encoder = json.loads((RAIZ / "configs/indice.json").read_text(encoding="utf-8"))["encoder"]

    anterior = RAIZ / "corpus_manifest.json"
    previo = json.loads(anterior.read_text(encoding="utf-8")) if anterior.is_file() else {}
    previo = previo if isinstance(previo, dict) else {}
    manifiesto = {
        "equipo": args.equipo or previo.get("equipo", ""),
        "licencia": LICENCIA,
        "fecha_generacion": date.today().isoformat(),
        "enlace_nube": args.enlace or previo.get("enlace_nube", ""),
        "encoder": encoder.get("repo_id"),
        "dimension": encoder.get("dimensiones"),
        "indice": "faiss.IndexFlatIP",
        "n_documentos": len(fichas_docs),
        "n_fragmentos": resumen["fragmentos"],
        "documentos": fichas_docs,
    }
    anterior.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")

    cobertura = cobertura_por_area()
    escribir_corpus_md(fichas_docs, resumen, cobertura, encoder)
    total, cubierto, _, _ = cobertura
    print(f"corpus_manifest.json: {len(fichas_docs)} documentos, {resumen['fragmentos']} fragmentos")
    print(f"CORPUS.md: cobertura ponderada por área escrita "
          f"({sum(cubierto.values())}/{sum(total.values())} menciones sumando áreas)")


if __name__ == "__main__":
    main()
