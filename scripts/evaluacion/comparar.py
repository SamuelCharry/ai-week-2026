"""Comparación entre corridas y entre versiones del sistema.

`comparar_corridas` sirve para la verificación en vivo del sábado: dos corridas de
la misma configuración deben devolver lo mismo, salvo los tiempos.

Las tablas por versión (03 vs 04 vs 05) están en `scripts.experimentos.v04`, que es
quien sabe leer las carpetas de resultados de cada experimento.
"""
import json

from scripts.evaluacion.entrega import CAMPOS_TIEMPO


def sin_tiempos(valor):
    """Copia sin los campos de tiempo, que cambian entre corridas idénticas."""
    if isinstance(valor, dict):
        return {k: sin_tiempos(v) for k, v in valor.items() if k not in CAMPOS_TIEMPO}
    if isinstance(valor, list):
        return [sin_tiempos(v) for v in valor]
    return valor


def comparar_corridas(primera, segunda):
    """Qué ids difieren entre dos corridas, ignorando los tiempos."""
    a = {json.dumps(r['id'], sort_keys=True): sin_tiempos(r) for r in primera}
    b = {json.dumps(r['id'], sort_keys=True): sin_tiempos(r) for r in segunda}
    diferencias = [json.loads(clave) for clave in sorted(set(a) | set(b)) if a.get(clave) != b.get(clave)]
    return {'iguales': not diferencias, 'ids_distintos': diferencias, 'campos_excluidos': sorted(CAMPOS_TIEMPO)}
