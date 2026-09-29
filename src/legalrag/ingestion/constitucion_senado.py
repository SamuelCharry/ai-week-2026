"""Vuelve a sembrar la Constitución desde la Secretaría General del Senado.

La copia anterior venía del Gestor Normativo de la Función Pública, que no está
entre las fuentes admitidas del reto. Senado publica la segunda edición corregida
en dieciséis páginas enlazadas.

El cambio es de procedencia, no de calidad: con el segmentador corregido las dos
versiones dejan los 380 artículos con unidad propia. Se comprueba antes de escribir.

El original anterior no se borra. Queda en data/raw/co_constitucion_1_1991/000.html
y solo deja de estar referenciado en el manifiesto, de modo que volver atrás es
restaurar data/raw/manifest.antes_constitucion.json.

Uso: python -m legalrag.ingestion.constitucion_senado [--solo-listar]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))

from legalrag.ingestion.ampliar import continuaciones_senado, descargar  # noqa: E402

DOC_ID = "co_constitucion_1_1991"
BASE = "http://www.secretariasenado.gov.co/senado/basedoc/constitucion_politica_1991.html"
FUENTE = "Secretaría General del Senado - Base documental"
TITULO = "Constitución Política de Colombia"
ARTICULOS_ESPERADOS = 380


def descargar_tramos():
    estado, url_final, tipo_contenido, contenido = descargar(BASE)
    if estado != 200:
        raise RuntimeError(f"La página base respondió HTTP {estado}")
    partes = [{"url": BASE, "url_final": url_final, "tipo_contenido": tipo_contenido, "contenido": contenido}]
    return partes + continuaciones_senado(BASE, contenido)


def comprobar(partes):
    """Texto concatenado, cuerpos que reconoce el evaluador y artículos con unidad propia."""
    from legalrag.preprocessing.ingesta import _unidades_norma, extraer_html
    from legalrag.evaluation.oficial import cargar_citaciones

    texto = ""
    for orden, parte in enumerate(partes):
        bloques, _ = extraer_html(parte["contenido"], {"archivo": f"tramo {orden}", "url": parte["url"]})
        for bloque in bloques:
            if texto:
                texto += "\n\n"
            texto += bloque["texto"]
    citas = cargar_citaciones(RAIZ)
    cuerpos = citas.bodies(citas.extract(texto[:12000]))
    unidades = _unidades_norma(texto)
    propios = {u["articulo"] for u in unidades
               if u["tipo"] == "articulo" and (u["articulo"] or "").isdigit()}
    faltan = [n for n in range(1, ARTICULOS_ESPERADOS + 1) if str(n) not in propios]
    return texto, cuerpos, faltan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-listar", action="store_true", help="comprueba y no escribe nada")
    args = ap.parse_args()

    ruta_manifiesto = RAIZ / "data/raw/manifest.json"
    manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    entrada = next((d for d in manifiesto if d["doc_id"] == DOC_ID), None)
    if entrada is None:
        raise RuntimeError(f"No está {DOC_ID} en el manifiesto")

    partes = descargar_tramos()
    texto, cuerpos, faltan = comprobar(partes)
    print(f"Tramos: {len(partes)} | caracteres: {len(texto)} | cuerpos reconocidos: {sorted(cuerpos)}")
    print(f"Artículos sin unidad propia: {len(faltan)} {faltan[:20]}")
    if ("constitucion", None, None) not in cuerpos:
        raise RuntimeError("El evaluador no reconoce la Constitución en el encabezado de esta versión")
    if faltan:
        raise RuntimeError(f"La versión de Senado deja {len(faltan)} artículos sin unidad propia")
    if args.solo_listar:
        print("Comprobaciones correctas. No se escribió nada.")
        return

    respaldo = RAIZ / "data/raw/manifest.antes_constitucion.json"
    if not respaldo.is_file():
        respaldo.write_bytes(ruta_manifiesto.read_bytes())
    archivos = []
    for orden, parte in enumerate(partes):
        archivo = f"{DOC_ID}/senado_{orden:03d}.html"
        destino = RAIZ / "data/raw" / archivo
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(parte["contenido"])
        archivos.append({"url": parte["url"], "url_final": parte["url_final"],
                         "sha256": hashlib.sha256(parte["contenido"]).hexdigest(),
                         "bytes": len(parte["contenido"]), "http_status": 200,
                         "content_type": parte["tipo_contenido"], "last_modified": None,
                         "encoding": "auto", "archivo": archivo})
    ahora = datetime.now(timezone(timedelta(hours=-5))).replace(microsecond=0)
    entrada.update(titulo=TITULO, fuente=FUENTE, url=BASE, archivos_raw=archivos,
                   fecha_consulta=ahora.date().isoformat(), fecha_descarga=ahora.isoformat(),
                   origen_ampliacion="v05_fuente_admitida")
    temporal = ruta_manifiesto.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    temporal.replace(ruta_manifiesto)
    print(f"Manifiesto actualizado: {DOC_ID} apunta a {len(archivos)} tramos de Senado.")
    print(f"El original anterior sigue en data/raw/{DOC_ID}/000.html, sin referenciar.")


if __name__ == "__main__":
    main()
