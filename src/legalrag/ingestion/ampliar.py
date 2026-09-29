"""Amplía el corpus con las normas del banco que faltan (seed_targets.json).

Genera la URL según la fuente oficial cuando existe un patrón estable:
    Corte Constitucional C y T          relatoria/{año}/c-207-19.htm
    Corte Constitucional SU             relatoria/{año}/su016-20.htm   (sin guion)
    Leyes y decretos                    secretariasenado.gov.co/senado/basedoc/ley_0979_2005.html
Las sentencias de la Corte Suprema (SL, SP, SC, STL...) y los acuerdos no tienen
patrón: quedan en `data/raw/fuentes_manuales_v05.csv` para completar la URL a mano.

Cada descarga se valida: HTTP 200, tamaño mínimo y que el texto contenga el
identificador esperado (evita páginas de error que responden 200). En los PDF el
identificador se busca sobre el texto extraído, con el extractor de la ingesta.

Senado corta las normas largas en varias páginas ("ley_0906_2004_pr001.html"). Se
sigue la cadena y cada tramo queda como un original más del mismo documento.

Escribe los originales en data/raw/<doc_id>/{000,001,...}.<ext> y agrega las entradas
a data/raw/manifest.json, con copia previa en data/raw/manifest.antes_v05.json.
No toca documentos existentes.

Uso: python -m legalrag.ingestion.ampliar [--solo-listar] [--max N]
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))

LICENCIA = ("Texto jurídico oficial: reproducción conforme al art. 41 de la Ley 23 de 1982. "
            "Los derechos sobre notas y edición de terceros no se relicencian.")
AGENTE = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai-week-2026 corpus (uso académico)"
AREA_SLUG = {
    "Derecho constitucional": "constitucional", "Derecho administrativo": "administrativo",
    "Derecho penal": "penal", "Derecho procesal": "procesal", "Derecho comercial y sociedades": "comercial",
    "Derecho civil": "civil", "Derecho de familia": "familia", "Derecho tributario": "tributario",
    "Derecho laboral": "laboral",
    "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]": "mercados",
}
SALAS_CC = {"C", "T", "SU"}  # los autos (A) usan otra ruta: relatoria/autos/
SALAS_CSJ = {"SL": "Sala de Casación Laboral", "SP": "Sala de Casación Penal", "SC": "Sala de Casación Civil",
             "STL": "Sala de Casación Laboral", "STC": "Sala de Casación Civil", "STP": "Sala de Casación Penal",
             "AC": "Sala de Casación Civil", "AL": "Sala de Casación Laboral", "AP": "Sala de Casación Penal"}


def _norm(texto):
    texto = unicodedata.normalize("NFD", texto.casefold())
    return re.sub(r"\s+", " ", "".join(c for c in texto if unicodedata.category(c) != "Mn"))


def faltantes(raiz=RAIZ):
    """Objetivos sin documento en el corpus: los del banco y los de la muestra pública.

    seed_targets.json solo cubre el banco de 992. La muestra pública de 50 cita
    normas que no están ahí (C-468 de 2024, SU-16 de 2020, SU-277 de 2025), así que
    se añaden los cuerpos de su fundamento. Solo se usa el nombre de la norma para
    saber qué texto oficial descargar: ni las preguntas ni las respuestas esperadas
    entran al corpus ni al índice.
    """
    from legalrag.evaluation.oficial import cargar_muestra
    from legalrag.ingestion.cobertura import auditar_banco, indice_corpus
    from legalrag.citations.evidencia import nombre_cuerpo

    indice = indice_corpus(raiz)
    citas, _, por_cuerpo, _ = indice
    objetivos = json.loads((Path(raiz) / "data/oficial/data/seed_targets.json")
                           .read_text(encoding="utf-8"))["documentos"]
    canon = {o["norma"]: tuple(o["canonico"]) for o in objetivos}
    filas = [{**b, "canonico": canon[b["norma"]], "origen": "seed_targets"}
             for b in auditar_banco(raiz, indice=indice) if not b["en_corpus"]]
    vistos = {f["canonico"] for f in filas}
    preguntas, _ = cargar_muestra(raiz)
    for pregunta in preguntas:
        for cita in citas.extract(pregunta.get("legal_basis") or ""):
            cuerpo = cita[:3]
            if cuerpo in vistos or por_cuerpo.get(cuerpo):
                continue
            vistos.add(cuerpo)
            filas.append({"norma": nombre_cuerpo(cuerpo) or " ".join(str(x) for x in cuerpo if x),
                          "items_del_banco": 0, "canonico": cuerpo, "en_corpus": False,
                          "areas": [pregunta["area"]] if pregunta.get("area") else [],
                          "donde_buscar": "fundamento de la muestra de 50", "origen": "muestra_50"})
    return filas


def plan_fuente(objetivo, canonico):
    """Devuelve una ficha de manifiesto (sin archivos) y la URL, o motivo si es manual."""
    tipo, numero, anio = canonico
    areas = sorted({AREA_SLUG.get(a, "otros") for a in objetivo["areas"]})
    base = {"vigencia": "por_verificar",
            "alcance_vigencia": "Se preservan señales de la fuente. No se certifica vigencia ni consolidación completa.",
            "redistribuir_raw": True, "edicion_con_anotaciones": False, "licencia_fuente": LICENCIA,
            "advertencias_preliminares_fuente": [], "actualizacion_declarada_fuente": [], "temas": [],
            "areas": areas, "origen_ampliacion": "v05_seed_targets"}
    if tipo == "jurisprudencia":
        sala, num = numero.split("-")
        n = int(num)
        if sala in SALAS_CC:
            # La relatoría escribe "c-207-19" y "t-243-18" con guion, pero las SU sin él:
            # "su016-20". Con guion devuelve la portada del sitio con estado 200.
            radical = f"{sala.lower()}{'' if sala == 'SU' else '-'}{n:03d}-{str(anio)[2:]}"
            url = f"https://www.corteconstitucional.gov.co/relatoria/{anio}/{radical}.htm"
            ficha = {"doc_id": f"sentencia_cc_{sala.lower()}{n:03d}_{anio}", "titulo": f"Sentencia {sala}-{n:03d} de {anio}",
                     "tipo": "sentencia", "numero": f"{sala}-{n:03d}", "anio": int(anio),
                     "organo_emisor": "Corte Constitucional", "fuente": "Corte Constitucional - Relatoría", **base}
            # La relatoría escribe "c-207/19", "c-207-19" y "su011/20": el guion
            # entre sala y número es opcional y el separador del año varía.
            esperado = [rf"\b{sala.lower()}\s*-?\s*0*{n}\s*(/|de|-)\s*(19|20)?{str(anio)[2:]}\b"]
            return ficha, url, esperado, None
        if sala in SALAS_CSJ:
            ficha = {"doc_id": f"sentencia_csj_{sala.lower()}{n}_{anio}", "titulo": f"Sentencia {sala}{n}-{anio}",
                     "tipo": "sentencia", "numero": f"{sala}{n}", "anio": int(anio),
                     "organo_emisor": f"Corte Suprema de Justicia - {SALAS_CSJ[sala]}",
                     "fuente": "Corte Suprema de Justicia - Relatoría", **base}
            return ficha, None, [rf"\b{sala.lower()}\s*-?\s*{n}\s*(-|de|/)\s*{anio}\b"], "Corte Suprema: sin patrón de URL"
        return None, None, None, f"sala desconocida {sala}"
    if tipo in ("ley", "decreto") and numero and anio:
        n = int(numero)
        url = f"http://www.secretariasenado.gov.co/senado/basedoc/{tipo}_{n:04d}_{anio}.html"
        ficha = {"doc_id": f"{tipo}_{n}_{anio}", "titulo": f"{tipo.capitalize()} {n} de {anio}", "tipo": tipo,
                 "numero": str(n), "anio": int(anio),
                 "organo_emisor": "Congreso de la República" if tipo == "ley" else "Presidencia de la República",
                 "fuente": "Secretaría General del Senado - Base documental", **base}
        # Senado intercala anotaciones en el encabezado: "DECRETO <LEY> 4334 DE 2008",
        # "LEY ESTATUTARIA 1909 DE 2018". Se admite ese relleno sin dígitos.
        return ficha, url, [rf"\b{tipo}\b[^\d]{{0,18}}0*{n}\s+de\s+{anio}\b"], None
    return None, None, None, f"tipo sin patrón: {tipo}"


def descargar(url, reintentos=2):
    peticion = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    for intento in range(reintentos + 1):
        try:
            with urllib.request.urlopen(peticion, timeout=60) as r:
                return r.status, r.geturl(), r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as error:
            return error.code, url, "", b""
        except (urllib.error.URLError, TimeoutError) as error:
            if intento == reintentos:
                raise
            time.sleep(3 * (intento + 1))


def _decodificar(contenido):
    for cod in ("utf-8", "windows-1252", "latin-1"):
        try:
            return contenido.decode(cod)
        except UnicodeDecodeError:
            continue
    return contenido.decode("latin-1", errors="replace")


def _plano(contenido):
    """Texto sin etiquetas y normalizado, para buscar el identificador."""
    return _norm(re.sub(r"<[^>]+>", " ", _decodificar(contenido)))


def _texto_pdf(contenido):
    """Texto del PDF con el mismo extractor de la ingesta."""
    from legalrag.preprocessing.ingesta import extraer_pdf

    bloques, _ = extraer_pdf(contenido)
    return _norm(" ".join(b["texto"] for b in bloques))


def continuaciones_senado(url, contenido, maximo=60):
    """URLs de las partes siguientes de un documento partido en Senado.

    Senado corta las normas largas en "ley_0906_2004_pr001.html" y cada página
    enlaza la siguiente. Sin seguir la cadena solo se guarda el primer tramo.
    """
    base, archivo = url.rsplit("/", 1)
    raiz_nombre = re.sub(r"(_pr\d+)?\.html?$", "", archivo, flags=re.I)
    patron = re.compile(re.escape(raiz_nombre) + r"_pr\d+\.html?", re.I)
    vistos, partes, actual = {archivo.lower()}, [], contenido
    while len(partes) < maximo:
        siguiente = next((n for n in patron.findall(_decodificar(actual)) if n.lower() not in vistos), None)
        if siguiente is None:
            return partes
        vistos.add(siguiente.lower())
        estado, url_final, tipo_contenido, actual = descargar(f"{base}/{siguiente}")
        if estado != 200 or len(actual) < 1000:
            return partes
        partes.append({"url": f"{base}/{siguiente}", "url_final": url_final,
                       "tipo_contenido": tipo_contenido, "contenido": actual})
        time.sleep(1.0)
    return partes


def validar(contenido, tipo_contenido, esperado):
    if len(contenido) < 5000:
        return False, "contenido demasiado corto"
    if "pdf" in tipo_contenido.lower() or contenido[:4] == b"%PDF":
        try:
            texto = _texto_pdf(contenido)
        except Exception as error:
            return False, f"PDF ilegible: {type(error).__name__}"
        if not any(re.search(p, texto) for p in esperado):
            return False, "no aparece el identificador esperado en el PDF"
        return True, "ok (pdf)"
    texto = _plano(contenido)[:60000]
    if not any(re.search(p, texto) for p in esperado):
        return False, "no aparece el identificador esperado"
    return True, "ok"


def registrar(raiz, ficha, partes, manifiesto):
    """Guarda los originales del documento y agrega su ficha al manifiesto.

    partes: lista de tramos en orden de lectura. El primero es la página base y
    los siguientes son las continuaciones "_prNNN" de Senado, si las hay.
    """
    archivos = []
    for orden, parte in enumerate(partes):
        contenido = parte["contenido"]
        ext = ".pdf" if contenido[:4] == b"%PDF" else ".html"
        archivo = f"{ficha['doc_id']}/{orden:03d}{ext}"
        destino = raiz / "data/raw" / archivo
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido)
        archivos.append({"url": parte["url"], "url_final": parte["url_final"],
                         "sha256": hashlib.sha256(contenido).hexdigest(), "bytes": len(contenido),
                         "http_status": 200, "content_type": parte["tipo_contenido"], "last_modified": None,
                         "encoding": "PDF" if ext == ".pdf" else "auto", "archivo": archivo})
    ahora = datetime.now(timezone(timedelta(hours=-5))).replace(microsecond=0)
    entrada = {**ficha, "url": partes[0]["url"], "fecha_consulta": ahora.date().isoformat(),
               "fecha_descarga": ahora.isoformat(), "archivos_raw": archivos}
    manifiesto.append(entrada)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-listar", action="store_true", help="no descarga; solo escribe el plan")
    ap.add_argument("--max", type=int, default=0, help="máximo de descargas (0 = todas)")
    args = ap.parse_args()

    ruta_manifiesto = RAIZ / "data/raw/manifest.json"
    manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    existentes = {d["doc_id"] for d in manifiesto}

    plan, manuales = [], []
    for objetivo in sorted(faltantes(RAIZ), key=lambda o: -o["items_del_banco"]):
        ficha, url, esperado, motivo = plan_fuente(objetivo, objetivo["canonico"])
        if ficha and ficha["doc_id"] in existentes:
            continue
        fila = {"norma": objetivo["norma"], "items_del_banco": objetivo["items_del_banco"],
                "doc_id": ficha["doc_id"] if ficha else "", "url": url or "", "motivo_manual": motivo or "",
                "donde_buscar": objetivo["donde_buscar"]}
        (manuales if motivo else plan).append((fila, ficha, esperado))

    # URLs completadas a mano en una ejecución anterior.
    ruta_manual = RAIZ / "data/raw/fuentes_manuales_v05.csv"
    motivos_previos = {}
    if ruta_manual.is_file():
        with ruta_manual.open(encoding="utf-8-sig") as f:
            anteriores = list(csv.DictReader(f))
        dadas = {r["norma"]: r["url"].strip() for r in anteriores if r.get("url", "").strip()}
        # El diagnóstico que dejó fuentes_csj es más concreto que "sin patrón de URL".
        motivos_previos = {r["norma"]: r.get("motivo_manual", "").strip() for r in anteriores
                           if r.get("motivo_manual", "").strip()
                           and not r["motivo_manual"].startswith(("Corte Suprema:", "tipo sin patrón"))}
        for fila, ficha, esperado in list(manuales):
            if fila["norma"] in dadas and ficha:
                fila["url"] = dadas[fila["norma"]]
                manuales.remove((fila, ficha, esperado))
                plan.append((fila, ficha, esperado))
    for fila, _, _ in manuales:
        fila["motivo_manual"] = motivos_previos.get(fila["norma"], fila["motivo_manual"])

    with ruta_manual.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["norma", "items_del_banco", "doc_id", "url", "motivo_manual", "donde_buscar"])
        w.writeheader()
        w.writerows(fila for fila, _, _ in manuales)
    print(f"Descargables por patrón o URL manual: {len(plan)} | pendientes manuales: {len(manuales)} "
          f"({sum(f['items_del_banco'] for f, _, _ in manuales)} menciones del banco) -> {ruta_manual.name}")
    if args.solo_listar:
        for fila, _, _ in plan:
            print(f"  {fila['items_del_banco']:3d}  {fila['norma']:32s} {fila['url']}")
        return

    respaldo = RAIZ / "data/raw/manifest.antes_v05.json"
    if not respaldo.is_file():
        respaldo.write_bytes(ruta_manifiesto.read_bytes())
    informe = []
    for n, (fila, ficha, esperado) in enumerate(plan, 1):
        if args.max and n > args.max:
            break
        try:
            estado, url_final, tipo_contenido, contenido = descargar(fila["url"])
            if estado != 200:
                resultado = f"HTTP {estado}"
            else:
                ok, detalle = validar(contenido, tipo_contenido, esperado)
                resultado = detalle
                if ok:
                    partes = [{"url": fila["url"], "url_final": url_final,
                               "tipo_contenido": tipo_contenido, "contenido": contenido}]
                    if "secretariasenado" in fila["url"]:
                        partes += continuaciones_senado(fila["url"], contenido)
                        if len(partes) > 1:
                            resultado = f"ok ({len(partes)} tramos)"
                    registrar(RAIZ, ficha, partes, manifiesto)
                    temporal = ruta_manifiesto.with_suffix(".json.tmp")
                    temporal.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
                    temporal.replace(ruta_manifiesto)
        except Exception as error:  # la red falla por documento; se registra y se sigue
            resultado = f"{type(error).__name__}: {error}"
        informe.append({**fila, "resultado": resultado})
        print(f"[{n}/{len(plan)}] {fila['norma']}: {resultado}", flush=True)
        time.sleep(1.0)
    with (RAIZ / "data/raw/ampliacion_v05.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(informe[0]) if informe else ["norma"])
        w.writeheader()
        w.writerows(informe)
    agregados = sum(r["resultado"].startswith("ok") for r in informe)
    print(f"Agregados al manifiesto: {agregados}. Informe: data/raw/ampliacion_v05.csv")


if __name__ == "__main__":
    main()
