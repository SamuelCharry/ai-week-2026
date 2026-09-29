"""Empaqueta los originales del corpus para compartirlos (Drive) antes de la ingesta.

Deja data/entrega/corpus_raw.zip con:

    data/raw/                       originales, textos OCR, manifest.json,
                                    reconstruccion.csv y grafo_*.csv (sin cachés)
    configs/corpus_objetivos.json   por qué entró cada documento
    configs/corpus_exclusiones.json por qué no entró lo que el grafo pedía, con evidencia
    reports/auditoria_corpus.*      último informe de scripts.corpus.auditoria
    LICENSE, LEEME.md, SHA256SUMS

Se niega a empaquetar si la última auditoría tiene hallazgos críticos o si el manifiesto
cambió después de ella: el ZIP que se comparte debe ser el auditado.

Uso: python -m scripts.corpus.empaquetar_raw [--nombre corpus_raw.zip]
"""
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())

LICENCIA = """Corpus de derecho colombiano - AI Week 2026, Universidad de los Andes

Los textos normativos y las providencias judiciales colombianas son de dominio público
(art. 41 de la Ley 23 de 1982). Se descargaron de publicadores oficiales del Estado
colombiano; la URL y la fecha de cada original están en data/raw/manifest.json.

La selección, el manifiesto, la reconstrucción, el OCR, el grafo normativo y demás
procesamiento hecho por el equipo se publican bajo Creative Commons Atribución 4.0
Internacional (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

Las notas de vigencia, concordancias y ediciones de terceros incluidas en algunas
fuentes (por ejemplo, las de Avance Jurídico en las compilaciones oficiales) no se
relicencian y conservan los derechos de sus autores.
"""


def sha256(ruta):
    digest = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def archivos():
    raw = RAIZ / "data/raw"
    for ruta in sorted(raw.rglob("*")):
        partes = ruta.relative_to(raw).parts
        if ruta.is_file() and not any(p.startswith(".") for p in partes):
            yield ruta
    for relativa in ("configs/corpus_objetivos.json", "configs/corpus_exclusiones.json",
                     "reports/auditoria_corpus.md", "reports/auditoria_corpus.csv", "reports/auditoria_corpus.json"):
        if (RAIZ / relativa).is_file():
            yield RAIZ / relativa


def leeme(manifiesto, auditoria, exclusiones):
    tipos = Counter("sentencias y autos" if d.get("tipo") in ("sentencia", "auto") else "normas" for d in manifiesto)
    origenes = Counter(d.get("origen_ampliacion") or "reconstruccion_v06" for d in manifiesto)
    ocr = sum(1 for d in manifiesto for a in d.get("archivos_raw", []) if a.get("texto_derivado"))
    fuentes = Counter(d.get("fuente") for d in manifiesto).most_common()
    motivos = Counter(e["motivo"] for e in exclusiones)
    niveles = Counter(d.get("nivel", "sin_clasificar") for d in manifiesto)
    vigencias = Counter(d.get("vigencia_fuente", "sin_clasificar") for d in manifiesto)
    return f"""# Corpus de derecho colombiano: originales

Generado el {datetime.now().isoformat(timespec='minutes')}. Contiene los originales
descargados de fuentes oficiales, sin procesar. La ingesta y el índice se corren aparte.

## Contenido

- **{len(manifiesto)} documentos**: {', '.join(f'{n} {t}' for t, n in tipos.items())}.
- {ocr} originales escaneados o en Word llevan su texto al lado (`NNN.ocr.txt` por OCR, `NNN.txt` desde .doc),
  con hash, método y motivo en el campo `texto_derivado` del manifiesto.
- Por qué entró cada uno (`origen_ampliacion` en el manifiesto): {', '.join(f'{o}: {n}' for o, n in origenes.most_common())}.
- {len(exclusiones)} referencias que el grafo pedía y no entraron, cada una con motivo y evidencia en
  `configs/corpus_exclusiones.json`: {', '.join(f'{m}: {n}' for m, n in motivos.items())}.

Fuentes: {'; '.join(f'{f} ({n})' for f, n in fuentes)}.

## Núcleo y complementario (contra el ruido en la recuperación)

Nada se borró, pero no todo debe ir al índice principal. Cada documento del manifiesto trae:

- `nivel`: {', '.join(f'{k}: {n}' for k, n in niveles.most_common())}.
  - `nucleo`: Constitución, códigos, normas no derogadas, jurisprudencia curada (seed, hitos,
    grafo, unificación, Sala Penal), circulares y doctrina curada.
  - `complementario`: normas derogadas o inexequibles, leyes aprobatorias de tratados,
    honoríficas o ambientales (campo `alcance`), y lo que entró por barrido completo
    (sentencias C/SU, decretos del Gestor) sin que el corpus lo cite en 4 documentos o más.
- `vigencia_fuente`: {', '.join(f'{k}: {n}' for k, n in vigencias.most_common())}. Sale de las
  anotaciones oficiales del Senado; las derogaciones parciales quedan en el texto de cada
  artículo ("<Artículo derogado por ...>") y la ingesta las conserva.

Recomendación: indexar el núcleo y medir con `scripts/evaluate.py` sobre la muestra si sumar
el complementario mejora o empeora. La doctrina masiva (conceptos de Supersociedades y de
Función Pública, ~37.000) no está incluida por el mismo motivo.

## Verificar

    sha256sum -c SHA256SUMS

Auditoría incluida (`reports/auditoria_corpus.md`): {auditoria['hallazgos']}.

## Usar

1. Descomprimir en la raíz del repositorio (crea `data/raw/`).
2. La ingesta debe ser la versión 0.7.0 o posterior de `scripts/corpus/ingesta.py`: lee los
   textos derivados (`texto_derivado`), retira la navegación y el pie editorial de Senado y
   segmenta compendios, conceptos y circulares por bloques. Con una versión anterior
   los escaneados salen ilegibles y el pie de Avance Jurídico queda pegado a los artículos.
3. `ejecutar_ingesta(RAIZ)` y `guardar_resultados(...)` (notebook e05, sección 2), luego el índice.
4. `python -m scripts.corpus.auditoria` otra vez después de la ingesta (sección F).

## Reconstruir desde las URL

`python -m scripts.corpus.reconstruir` vuelve a descargar todo desde las URL del manifiesto
y del inventario, eligiendo por documento la fuente más completa.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", default="corpus_raw.zip")
    ap.add_argument("--parcial", action="store_true",
                    help="entrega intermedia: admite el grafo abierto (y lo declara en LEEME.md)")
    args = ap.parse_args()

    ruta_manifiesto = RAIZ / "data/raw/manifest.json"
    ruta_auditoria = RAIZ / "reports/auditoria_corpus.json"
    if not ruta_auditoria.is_file():
        sys.exit("Falta la auditoría: correr python -m scripts.corpus.auditoria")
    auditoria = json.loads(ruta_auditoria.read_text(encoding="utf-8"))
    criticos = [c for c in auditoria.get("por_comprobacion", []) if c["severidad"] == "CRITICO"]
    # --parcial tolera solo el grafo abierto: una entrega intermedia puede salir con referencias
    # citadas aún por incorporar (listadas en grafo_faltantes.csv), nunca con originales
    # alterados, faltantes, fugas o manifiesto incompleto.
    bloqueantes = [c for c in criticos if not (args.parcial and c["comprobacion"] == "grafo_abierto")]
    if bloqueantes:
        sys.exit(f"La auditoría tiene hallazgos críticos {bloqueantes}: no se empaqueta")
    grafo_abierto = sum(c["cantidad"] for c in criticos if c["comprobacion"] == "grafo_abierto")
    if ruta_manifiesto.stat().st_mtime > ruta_auditoria.stat().st_mtime:
        sys.exit("El manifiesto cambió después de la auditoría: volver a auditar antes de empaquetar")
    manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    ruta_excl = RAIZ / "configs/corpus_exclusiones.json"
    exclusiones = json.loads(ruta_excl.read_text(encoding="utf-8"))["exclusiones"] if ruta_excl.is_file() else []

    destino = RAIZ / "data/entrega" / args.nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".zip.part")
    sumas = []
    with zipfile.ZipFile(temporal, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for ruta in archivos():
            relativa = ruta.relative_to(RAIZ).as_posix()
            z.write(ruta, relativa)
            sumas.append(f"{sha256(ruta)}  {relativa}")
        z.writestr("LICENSE", LICENCIA)
        z.writestr("LEEME.md", leeme(manifiesto, auditoria, exclusiones) + (
            f"\n## Entrega parcial\n\nEl grafo normativo sigue abierto: {grafo_abierto} referencias citadas por 4 "
            "documentos o más aún no están en el corpus ni excluidas (lista en `data/raw/grafo_faltantes.csv`; "
            "detalle en `reports/auditoria_corpus.csv`, comprobación `grafo_abierto`). La siguiente entrega las "
            "incorpora o las excluye con evidencia.\n" if grafo_abierto else ""))
        z.writestr("SHA256SUMS", "\n".join(sumas) + "\n")
    temporal.replace(destino)
    print(f"{destino.relative_to(RAIZ)}: {len(sumas)} archivos, {destino.stat().st_size / 2**20:.0f} MiB, "
          f"sha256 {sha256(destino)}")


if __name__ == "__main__":
    main()
