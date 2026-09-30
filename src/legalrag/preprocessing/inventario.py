"""Inventario evaluable a partir de una preparación (preprocessing.preparar_corpus).

Mismo formato que `ingestion.fijar_corpus_evaluacion` (corpus_manifest.json, excluidos.json,
restricciones.json y snapshot.json con su snapshot_id), sin exigir los reportes de perfilado.
Entran los documentos extraídos y aptos para búsqueda, con su texto y SHA-256.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def fijar_inventario(preparado, destino, raiz):
    """Inventario evaluable: los documentos extraídos y aptos para búsqueda, con texto y SHA-256."""
    campos = ("doc_id", "titulo", "tipo", "numero", "anio", "fuente", "url", "fecha_consulta", "organo_emisor",
              "areas", "temas", "nivel", "vigencia", "vigencia_fuente", "licencia_fuente", "redistribuir_raw",
              "edicion_con_anotaciones", "origen_ampliacion", "areas_por_epigrafe", "sha256_texto", "caracteres",
              "estado_extraccion", "avisos")
    elegidos, excluidos, restricciones = [], [], []
    for linea in (preparado / "documentos.jsonl").read_text(encoding="utf-8").splitlines():
        registro = json.loads(linea)
        fila = {k: registro.get(k) for k in campos}
        fila["texto_archivo"] = (Path(preparado).relative_to(raiz) / (registro.get("texto_archivo") or "")).as_posix()
        fila["restricciones_especificas"] = sorted(set(registro.get("pendientes_preparacion", [])))
        fila["archivos_raw"] = registro["archivos_raw"]
        if registro["estado_extraccion"] == "error" or registro.get("apta_para_busqueda") is False:
            excluidos.append(fila)
        else:
            elegidos.append(fila)
            if fila["restricciones_especificas"]:
                restricciones.append({"doc_id": fila["doc_id"], "restricciones": fila["restricciones_especificas"]})
    destino.mkdir(parents=True, exist_ok=True)
    archivos = []
    for nombre, contenido in (("corpus_manifest.json", sorted(elegidos, key=lambda r: r["doc_id"])),
                              ("excluidos.json", sorted(excluidos, key=lambda r: r["doc_id"])),
                              ("restricciones.json", restricciones)):
        ruta = destino / nombre
        ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        archivos.append({"archivo": nombre, "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()})
    snapshot = {"version": destino.name,
                "snapshot_id": hashlib.sha256(json.dumps(archivos, sort_keys=True).encode()).hexdigest(),
                "fecha_utc": datetime.now(timezone.utc).isoformat(),
                "tipo_snapshot": "desde_data_raw_con_src_main_sin_perfilado",
                "archivos_version": archivos, "documentos_evaluables": len(elegidos),
                "documentos_excluidos": len(excluidos)}
    (destino / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  inventario: {len(elegidos)} documentos evaluables, {len(excluidos)} excluidos -> {destino}")
