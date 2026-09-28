"""Resuelve la URL de las providencias de la Corte Suprema en su relatoría.

La relatoría no expone un patrón de URL: el nombre del archivo lleva el radicado y
la carpeta del boletín cambia de un mes a otro, con varias formas de escribirla.

    relatorias/pe/b1ago2019/SP1945-2019(50523).PDF
    relatorias/la/bnov2022/SL3385-2022.pdf
    relatorias/la/bmar2018/SL648-2018.doc

Ninguna de las tres se puede construir a partir del identificador. El portal sí
indexa cada providencia como una entrada con el enlace a su archivo, así que se
consulta ese índice y se lee el enlace de ahí. Solo se aceptan PDF: la ingesta no
procesa documentos de Word.

Completa la columna `url` de data/raw/fuentes_manuales_v05.csv y deja anotado en
`motivo_manual` lo que no se pudo resolver, para revisarlo a mano.

Uso: python -m scripts.corpus.fuentes_csj [--solo-listar]
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ))

INDICE = "https://cortesuprema.gov.co/corte/index.php/wp-json/wp/v2/posts"
SUBIDAS = "https://www.cortesuprema.gov.co/corte/wp-content/uploads"
AGENTE = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai-week-2026 corpus (uso académico)"
ARCHIVO_RELATORIA = re.compile(r"https?://[^\s\"']*/relatorias/[^\s\"']+", re.I)

# Carpeta de cada sala dentro de relatorias/ y páginas donde la relatoría enlaza
# sus boletines, de donde se leen los nombres reales de las carpetas por mes.
SALA_CARPETA = {"SL": "la", "AL": "la", "STL": "la", "SP": "pe", "AP": "pe", "STP": "pe",
                "SC": "ci", "AC": "ci", "STC": "ci"}
INDICES_BOLETIN = {
    "la": ("sala-laboral-relatoria", "sala-de-casacion-laboral-relatoria-boletines-historico"),
    "pe": ("sala-de-casacion-penal-relatoria", "sala-de-casacion-penal-relatoria-boletines"),
    "ci": ("sala-civil-relatoria",),
}
_carpetas = {}


def identificador(norma):
    """"Sentencia SL-3385 de 2022" -> "SL3385-2022", como lo nombra la relatoría."""
    m = re.search(r"\b([A-Z]{2,3})\s*-?\s*(\d{1,5})\s*(?:de|/|-)\s*(\d{4})\b", norma, re.I)
    return f"{m.group(1).upper()}{int(m.group(2))}-{m.group(3)}" if m else None


def consultar(identificador, reintentos=2):
    """Entradas del índice del portal que mencionan el identificador."""
    url = f"{INDICE}?search={urllib.parse.quote(identificador)}&per_page=5"
    peticion = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    for intento in range(reintentos + 1):
        try:
            with urllib.request.urlopen(peticion, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError:
            return []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if intento == reintentos:
                return []
            time.sleep(3 * (intento + 1))
    return []


def _pedir(url, binario=False):
    peticion = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    try:
        with urllib.request.urlopen(peticion, timeout=60) as r:
            return r.status, r.read() if binario else r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, b"" if binario else ""
    except (urllib.error.URLError, TimeoutError):
        return 0, b"" if binario else ""


def carpetas_boletin(sala):
    """Carpetas de boletín que la relatoría enlaza, leídas de sus propios índices.

    Los nombres no siguen una sola regla ("bnov2022", "b1may2022", "b82019"), así
    que se toman de la página en vez de construirlos.
    """
    if sala in _carpetas:
        return _carpetas[sala]
    nombres = []
    for pagina in INDICES_BOLETIN.get(sala, ()):
        estado, html = _pedir(f"https://cortesuprema.gov.co/{pagina}/")
        if estado != 200:
            continue
        nombres += re.findall(rf"relatorias/{sala}/([A-Za-z0-9]+)/", html, re.I)
    _carpetas[sala] = list(dict.fromkeys(nombres))
    return _carpetas[sala]


def buscar_en_boletines(ident, sala, anio):
    """Prueba el archivo dentro de las carpetas de boletín del año y el siguiente.

    En laboral el archivo se llama como la providencia ("SL3385-2022.pdf"); en las
    otras salas suele llevar el radicado y entonces no hay nada que probar.
    """
    años = {str(anio), str(int(anio) + 1)}
    for carpeta in [c for c in carpetas_boletin(sala) if any(c.endswith(a) for a in años)]:
        for extension in ("pdf", "PDF"):
            url = f"{SUBIDAS}/relatorias/{sala}/{carpeta}/{ident}.{extension}"
            estado, contenido = _pedir(url, binario=True)
            if estado == 200 and contenido[:4] == b"%PDF":
                return url
            time.sleep(0.2)
    return None


def resolver(norma):
    """Devuelve (url, motivo). La URL es None si no hay un PDF utilizable."""
    ident = identificador(norma)
    if not ident:
        return None, "no se pudo leer el identificador de la norma"
    encontrados = []
    for entrada in consultar(ident):
        texto = json.dumps(entrada, ensure_ascii=False)
        for candidata in ARCHIVO_RELATORIA.findall(texto.replace("\\/", "/")):
            candidata = candidata.rstrip("\\\"' ")
            nombre = candidata.rsplit("/", 1)[-1]
            if ident.lower() in urllib.parse.unquote(nombre).lower():
                encontrados.append(candidata)
    pdf = [u for u in dict.fromkeys(encontrados) if u.lower().endswith(".pdf")]
    if pdf:
        return pdf[0], "ok"
    sala, anio = ident.rstrip("0123456789-").upper(), ident.rsplit("-", 1)[1]
    por_carpeta = buscar_en_boletines(ident, SALA_CARPETA.get(sala, ""), anio)
    if por_carpeta:
        return por_carpeta, "ok"
    if encontrados:
        formatos = sorted({u.rsplit(".", 1)[-1].lower() for u in encontrados})
        return None, f"la relatoría solo publica {', '.join(formatos)}: {encontrados[0]}"
    return None, "no está como archivo propio en la relatoría (solo dentro del boletín)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-listar", action="store_true", help="no escribe el CSV")
    args = ap.parse_args()

    ruta = RAIZ / "data/raw/fuentes_manuales_v05.csv"
    with ruta.open(encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
        campos = list(filas[0]) if filas else []
    resueltas = 0
    for fila in filas:
        if fila.get("url", "").strip():
            continue
        url, motivo = resolver(fila["norma"])
        if url:
            fila["url"], resueltas = url, resueltas + 1
        else:
            fila["motivo_manual"] = motivo
        print(f"{fila['norma']:32s} {url or motivo}", flush=True)
        time.sleep(1.0)
    if not args.solo_listar:
        with ruta.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
    print(f"\nResueltas: {resueltas} de {len(filas)}. Archivo: {ruta.name}")


if __name__ == "__main__":
    main()
