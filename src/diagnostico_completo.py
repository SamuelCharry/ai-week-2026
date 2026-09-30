#!/usr/bin/env python3
"""Diagnóstico completo del sistema LegalRAG sobre las 50 preguntas de muestra.

Analiza las respuestas generadas y clasifica TODOS los problemas:
  - Cerradas: abstenciones, errores del modelo, fallas del corpus
  - Texto libre (semi_open / open_ended): calidad de citas, abstenciones, cobertura
  - Citas globales: qué normas se citan sin respaldo y cuáles se pierden
  - Recuperación: qué tan bien el sistema trae la evidencia relevante
  - Abstención: patrones de cuándo se abstiene bien/mal
  - Prompt / JSON: problemas de parsing, truncado, campos vacíos
  - Estimación de impacto: cuántos puntos se ganan arreglando cada problema

Uso (desde la raíz del repo, con el entorno virtual activado):
    python src/diagnostico_completo.py

Requisito: haber corrido antes:
    python -m legalrag answer --split sample
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "data/oficial/scripts"))

try:
    import citations
except ImportError:
    sys.exit("No se pudo importar el módulo citations del evaluador oficial.\n"
             "Asegúrate de estar en la raíz del repo y de tener data/oficial/scripts/citations.py")

# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────
MUESTRA = RAIZ / "data/oficial/data/sample_50.jsonl"
ENTREGA = RAIZ / "data/reproduccion/submissions_sample.jsonl"
DETALLE_DIR = RAIZ / "data/reproduccion/submissions_sample_respuestas"
RESUMEN = RAIZ / "data/reproduccion/submissions_sample_resumen.json"


def leer_jsonl(ruta):
    if not ruta.is_file():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def leer_detalle(qid):
    ruta = DETALLE_DIR / f"{qid}.json"
    if ruta.is_file():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades del evaluador (reimplementadas para no depender de evaluate.py)
# ──────────────────────────────────────────────────────────────────────────────
MAX_PASAJES = 10
W_SIN_RESPALDO = 0.5
W_ABSTENCION = 0.5
PTS = {"cerradas": 20.0, "ragas": 30.0, "citas": 20.0, "abstencion": 10.0}


def answer_text(sub):
    f = sub.get("formato")
    if f == "multiple_choice":
        return sub.get("justificacion") or ""
    if f == "semi_open":
        return " ".join(str(sub.get(k) or "") for k in ("respuesta", "referencia_legal"))
    return " ".join(str(sub.get(k) or "") for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion"))


def citas_en_pasajes(sub):
    cites = set()
    for p in (sub.get("pasajes_recuperados") or [])[:MAX_PASAJES]:
        cites |= citations.extract(str(p.get("texto") or ""))
    return cites


# ──────────────────────────────────────────────────────────────────────────────
# Secciones del diagnóstico
# ──────────────────────────────────────────────────────────────────────────────
def seccion(titulo):
    ancho = 80
    print(f"\n{'═' * ancho}")
    print(f"  {titulo}")
    print(f"{'═' * ancho}")


def subseccion(titulo):
    print(f"\n── {titulo} {'─' * max(0, 74 - len(titulo))}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not MUESTRA.is_file():
        sys.exit(f"No se encuentra la muestra: {MUESTRA}")
    if not ENTREGA.is_file():
        sys.exit(f"No se encuentra la entrega: {ENTREGA}\n"
                 "Ejecuta primero:  python -m legalrag answer --split sample")

    clave = {p["id"]: p for p in leer_jsonl(MUESTRA)}
    entrega = {r["id"]: r for r in leer_jsonl(ENTREGA)}
    ids_cerradas = {qid for qid, k in clave.items() if k["formato"] == "multiple_choice"}
    ids_semi = {qid for qid, k in clave.items() if k["formato"] == "semi_open"}
    ids_open = {qid for qid, k in clave.items() if k["formato"] == "open_ended"}

    print(f"Preguntas en la muestra:  {len(clave)}")
    print(f"  Cerradas:        {len(ids_cerradas)}")
    print(f"  Semiabiertas:    {len(ids_semi)}")
    print(f"  Abiertas:        {len(ids_open)}")
    print(f"Respuestas entregadas:    {len(entrega)}")
    faltantes = set(clave) - set(entrega)
    if faltantes:
        print(f"  ⚠ Preguntas sin respuesta ({len(faltantes)}): {sorted(faltantes)}")

    # ──────────────────────────────────────────────────────────────────────────
    # 0. Resumen de latencia (si existe)
    # ──────────────────────────────────────────────────────────────────────────
    if RESUMEN.is_file():
        res = json.loads(RESUMEN.read_text(encoding="utf-8"))
        seccion("0. LATENCIA")
        presupuesto = res.get("presupuesto_segundos", "?")
        promedio = res.get("segundos_promedio")
        p95 = res.get("segundos_p95")
        if promedio is not None:
            print(f"  Promedio: {promedio:.1f} s   P95: {p95:.1f} s   Presupuesto: {presupuesto} s")
            if promedio > float(presupuesto or 999):
                print(f"  ⚠ El promedio SUPERA el presupuesto de {presupuesto} s")
        errores = res.get("errores")
        if errores:
            print(f"  ⚠ Errores en la ejecución ({len(errores)}):")
            for e in errores[:10]:
                print(f"    id={e['id']}: {e['error']}: {e['detalle'][:100]}")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Problemas del pipeline (JSON, abstenciones, campos vacíos)
    # ──────────────────────────────────────────────────────────────────────────
    seccion("1. PROBLEMAS DEL PIPELINE (JSON, abstenciones, campos vacíos)")

    problemas_pipeline = defaultdict(list)
    campos_vacios_por_formato = defaultdict(lambda: defaultdict(int))
    latencias = {}

    CAMPOS = {
        "multiple_choice": ["respuesta_correcta", "justificacion", "descarte_opciones"],
        "semi_open": ["respuesta", "palabras_clave", "referencia_legal"],
        "open_ended": ["marco_normativo", "analisis", "jurisprudencia", "conclusion"],
    }

    for qid in sorted(clave):
        det = leer_detalle(qid)
        problema = det.get("problema")
        seg = det.get("segundos")
        if seg is not None:
            latencias[qid] = seg

        s = entrega.get(qid)
        if s is None:
            problemas_pipeline["sin_respuesta"].append(qid)
            continue

        if problema:
            problemas_pipeline[problema].append(qid)

        fmt = s.get("formato", "?")
        if not s.get("abstencion"):
            for campo in CAMPOS.get(fmt, []):
                val = s.get(campo)
                if val is None or val == "" or val == [] or val == {}:
                    campos_vacios_por_formato[fmt][campo] += 1
                    problemas_pipeline[f"campo_vacío:{fmt}.{campo}"].append(qid)

    if problemas_pipeline:
        conteo = Counter()
        for tipo, ids in sorted(problemas_pipeline.items()):
            conteo[tipo] = len(ids)
            print(f"  {tipo:50s}  {len(ids):3d}  ids={ids[:8]}")
        print(f"\n  Total de problemas detectados por el pipeline: {sum(conteo.values())}")
    else:
        print("  ✓ No se detectaron problemas de pipeline en los archivos de detalle.")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. CERRADAS — Accuracy detallada
    # ──────────────────────────────────────────────────────────────────────────
    seccion("2. CERRADAS — Accuracy detallada")

    print(f"\n  {'id':>5}  {'clave':5} {'resp':5} {'abst':5} {'norma_recup':12} {'veredicto':12} {'problema'}")
    print(f"  {'─'*5}  {'─'*5} {'─'*5} {'─'*5} {'─'*12} {'─'*12} {'─'*30}")

    resultado_cerradas = Counter()
    fallos_cerradas = []

    for qid in sorted(ids_cerradas):
        k = clave[qid]
        s = entrega.get(qid, {})
        det = leer_detalle(qid)
        problema = det.get("problema", "?")

        ref_citas = citations.bodies(citations.extract(k.get("legal_basis") or ""))
        recuperadas = set()
        for p in (s.get("pasajes_recuperados") or [])[:MAX_PASAJES]:
            recuperadas |= citations.bodies(citations.extract(str(p.get("texto") or "")))
        norma_recup = "sin cita ref" if not ref_citas else ("sí" if ref_citas & recuperadas else "NO")

        abst = bool(s.get("abstencion"))
        resp = str(s.get("respuesta_correcta", "—"))

        if not abst and resp == k["respuesta_correcta"]:
            veredicto = "acierto"
        elif abst:
            veredicto = "abstencion"
        elif not ref_citas:
            veredicto = "sin_cita_ref"
        elif norma_recup == "NO":
            veredicto = "corpus"
        else:
            veredicto = "modelo"
        resultado_cerradas[veredicto] += 1

        marca = "  " if veredicto == "acierto" else "▸ "
        print(f"{marca}{qid:>5}  {k['respuesta_correcta']:5} {resp:5} {str(abst):5} "
              f"{norma_recup:12} {veredicto:12} {problema}")
        if veredicto != "acierto":
            fallos_cerradas.append({
                "id": qid, "clave": k["respuesta_correcta"], "respuesta": resp,
                "abstencion": abst, "norma_recuperada": norma_recup,
                "veredicto": veredicto, "problema": problema,
                "pregunta": k["pregunta"][:120],
                "legal_basis": k.get("legal_basis", "")[:120],
            })

    acc = resultado_cerradas["acierto"] / len(ids_cerradas) if ids_cerradas else 0
    print(f"\n  Resumen cerradas: {json.dumps(dict(resultado_cerradas), ensure_ascii=False)}")
    print(f"  Accuracy: {resultado_cerradas['acierto']}/{len(ids_cerradas)} = {acc:.1%}")
    print(f"  Puntos:   {PTS['cerradas'] * acc:.1f} / {PTS['cerradas']}")

    if fallos_cerradas:
        subseccion("Detalle de fallos en cerradas")
        for f in fallos_cerradas:
            print(f"\n  id={f['id']}  clave={f['clave']}  respondió={f['respuesta']}  "
                  f"veredicto={f['veredicto']}  problema={f['problema']}")
            print(f"    Pregunta:    {f['pregunta']}")
            print(f"    Legal basis: {f['legal_basis']}")
            print(f"    Norma recuperada: {f['norma_recuperada']}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. CITAS — Análisis global (todos los formatos)
    # ──────────────────────────────────────────────────────────────────────────
    seccion("3. CITAS — Análisis global")

    total_citas = defaultdict(int)
    problemas_citas = []
    citas_sin_respaldo_global = Counter()
    normas_no_encontradas_global = Counter()

    for qid in sorted(clave):
        k = clave[qid]
        s = entrega.get(qid, {})
        ref_citas = citations.extract(k.get("legal_basis") or "")
        if not ref_citas:
            continue  # no se evalúa

        total_citas["items_evaluados"] += 1

        if s.get("abstencion") or not s:
            total_citas["abstenciones"] += 1
            total_citas["n_ref_perdidas_abstencion"] += len(citations.bodies(ref_citas))
            continue

        respaldo = citas_en_pasajes(s)
        respuesta_citas = citations.extract(answer_text(s))

        ref_b = citations.bodies(ref_citas)
        got_b = citations.bodies(respuesta_citas)
        resp_b = citations.bodies(respaldo)

        aciertos = got_b & ref_b
        aciertos_con_respaldo = aciertos & resp_b
        aciertos_sin_respaldo = aciertos - resp_b
        fuera = got_b - ref_b
        sin_respaldo = fuera - resp_b

        total_citas["n_ref"] += len(ref_b)
        total_citas["n_citadas"] += len(got_b)
        total_citas["aciertos"] += len(aciertos)
        total_citas["aciertos_respaldados"] += len(aciertos_con_respaldo)
        total_citas["aciertos_sin_respaldo"] += len(aciertos_sin_respaldo)
        total_citas["incorrectas"] += len(fuera - sin_respaldo)
        total_citas["citas_sin_respaldo"] += len(sin_respaldo)

        # Normas de referencia no encontradas
        perdidas = ref_b - got_b
        for c in perdidas:
            normas_no_encontradas_global[c] += 1

        # Citas sin respaldo
        for c in sin_respaldo:
            citas_sin_respaldo_global[c] += 1

        if sin_respaldo or perdidas:
            problemas_citas.append({
                "id": qid, "formato": k["formato"],
                "sin_respaldo": [str(c) for c in sorted(sin_respaldo)],
                "perdidas": [str(c) for c in sorted(perdidas)],
                "ref_total": len(ref_b), "acertadas": len(aciertos),
            })

    recall = (total_citas["aciertos_respaldados"] + W_SIN_RESPALDO * total_citas["aciertos_sin_respaldo"]) / total_citas["n_ref"] if total_citas["n_ref"] else 0
    tasa_sr = total_citas["citas_sin_respaldo"] / total_citas["n_citadas"] if total_citas["n_citadas"] else 0
    indice = max(0, recall - 2 * tasa_sr)
    puntos_citas = PTS["citas"] * indice

    print(f"  Items evaluados (con legal_basis):  {total_citas['items_evaluados']}")
    print(f"  Abstenciones (no aportan citas):    {total_citas['abstenciones']}")
    print(f"  Normas de referencia totales:        {total_citas['n_ref']}")
    print(f"  Normas citadas en respuestas:        {total_citas['n_citadas']}")
    print(f"  Aciertos (norma correcta):           {total_citas['aciertos']}")
    print(f"    Con respaldo en pasajes:            {total_citas['aciertos_respaldados']}")
    print(f"    Sin respaldo (valen ×0.5):          {total_citas['aciertos_sin_respaldo']}")
    print(f"  Incorrectas (norma no es del ref):   {total_citas['incorrectas']}")
    print(f"  Citas sin respaldo (penalizan ×2):   {total_citas['citas_sin_respaldo']}")
    print(f"\n  recall_ponderado = {recall:.4f}")
    print(f"  tasa_sin_respaldo = {tasa_sr:.4f}")
    print(f"  índice = max(0, {recall:.4f} − 2 × {tasa_sr:.4f}) = {indice:.4f}")
    print(f"  Puntos: {puntos_citas:.1f} / {PTS['citas']}")

    if citas_sin_respaldo_global:
        subseccion("Citas sin respaldo más frecuentes (penalizan)")
        for cita, n in citas_sin_respaldo_global.most_common(15):
            print(f"  [{n}×] {cita}")

    if normas_no_encontradas_global:
        subseccion("Normas de referencia no citadas más frecuentes (recall perdido)")
        for norma, n in normas_no_encontradas_global.most_common(15):
            print(f"  [{n}×] {norma}")

    if problemas_citas:
        subseccion("Detalle por pregunta con problemas de citas")
        for p in problemas_citas[:20]:
            print(f"  id={p['id']} ({p['formato']})  acertadas={p['acertadas']}/{p['ref_total']}")
            if p["sin_respaldo"]:
                print(f"    SIN RESPALDO: {', '.join(p['sin_respaldo'][:5])}")
            if p["perdidas"]:
                print(f"    NO CITADAS:   {', '.join(p['perdidas'][:5])}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. ABSTENCIÓN — Calibración detallada
    # ──────────────────────────────────────────────────────────────────────────
    seccion("4. ABSTENCIÓN — Calibración detallada")

    abst_tabla = []
    tp = fp = fn = tn = 0

    for qid in sorted(clave):
        k = clave[qid]
        ref_citas = citations.extract(k.get("legal_basis") or "")
        # El evaluador excluye preguntas de texto libre sin legal_basis
        if qid not in ids_cerradas and not ref_citas:
            continue

        s = entrega.get(qid)
        abst = bool(s and s.get("abstencion"))

        if s is None:
            cat = "fn (sin respuesta)"
            fn += 1
            abst_tabla.append({"id": qid, "formato": k["formato"], "abstencion": "—",
                                "acerto": "—", "categoria": cat})
            continue

        if qid in ids_cerradas:
            acerto = s.get("respuesta_correcta") == k.get("respuesta_correcta")
        else:
            got = citations.extract(answer_text(s))
            acerto = bool(citations.bodies(ref_citas) & citations.bodies(got))

        if abst and not acerto:
            cat = "tp (abstuvo bien)"
            tp += 1
        elif abst and acerto:
            cat = "fp (abstuvo de más)"
            fp += 1
        elif not abst and acerto:
            cat = "tn (respondió bien)"
            tn += 1
        else:
            cat = "fn (respondió mal)"
            fn += 1

        abst_tabla.append({"id": qid, "formato": k["formato"], "abstencion": abst,
                            "acerto": acerto, "categoria": cat})

    total_abst = tp + fp + tn + fn
    calibracion = (tn + W_ABSTENCION * (tp + fp)) / total_abst if total_abst else 0

    print(f"  Items evaluados:         {total_abst}")
    print(f"  Respondió bien (tn):     {tn}")
    print(f"  Abstuvo bien (tp):       {tp}")
    print(f"  Abstuvo de más (fp):     {fp}")
    print(f"  Respondió mal (fn):      {fn}")
    print(f"  Calibración: ({tn} + 0.5 × ({tp} + {fp})) / {total_abst} = {calibracion:.4f}")
    print(f"  Puntos: {PTS['abstencion'] * calibracion:.1f} / {PTS['abstencion']}")

    subseccion("Preguntas respondidas MAL (fn) — las que más puntaje pierden")
    for row in abst_tabla:
        if row["categoria"].startswith("fn"):
            det = leer_detalle(row["id"])
            print(f"  id={row['id']:>5}  {row['formato']:20s}  problema={det.get('problema', '?')}")

    subseccion("Preguntas donde se abstuvo de más (fp) — respondía bien pero se abstuvo")
    for row in abst_tabla:
        if row["categoria"].startswith("fp"):
            det = leer_detalle(row["id"])
            print(f"  id={row['id']:>5}  {row['formato']:20s}  problema={det.get('problema', '?')}")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. RECUPERACIÓN — Cobertura de las normas de referencia
    # ──────────────────────────────────────────────────────────────────────────
    seccion("5. RECUPERACIÓN — ¿El corpus tiene las normas y se recuperan?")

    recup_ok = recup_parcial = recup_no = recup_na = 0
    fallos_recuperacion = []

    for qid in sorted(clave):
        k = clave[qid]
        ref_citas = citations.bodies(citations.extract(k.get("legal_basis") or ""))
        if not ref_citas:
            recup_na += 1
            continue

        s = entrega.get(qid, {})
        recuperadas = set()
        for p in (s.get("pasajes_recuperados") or [])[:MAX_PASAJES]:
            recuperadas |= citations.bodies(citations.extract(str(p.get("texto") or "")))

        encontradas = ref_citas & recuperadas
        if encontradas == ref_citas:
            recup_ok += 1
        elif encontradas:
            recup_parcial += 1
            fallos_recuperacion.append({
                "id": qid, "formato": k["formato"],
                "encontradas": len(encontradas), "total": len(ref_citas),
                "faltantes": [str(c) for c in sorted(ref_citas - recuperadas)],
            })
        else:
            recup_no += 1
            fallos_recuperacion.append({
                "id": qid, "formato": k["formato"],
                "encontradas": 0, "total": len(ref_citas),
                "faltantes": [str(c) for c in sorted(ref_citas)],
            })

    total_recup = recup_ok + recup_parcial + recup_no
    print(f"  Items con legal_basis evaluable:  {total_recup}")
    print(f"  ✓ Todas las normas recuperadas:   {recup_ok} ({recup_ok/total_recup:.0%})")
    print(f"  ~ Parcialmente recuperadas:        {recup_parcial} ({recup_parcial/total_recup:.0%})")
    print(f"  ✗ Ninguna norma recuperada:        {recup_no} ({recup_no/total_recup:.0%})")
    print(f"  — Sin cita extraíble en ref:       {recup_na}")

    if fallos_recuperacion:
        subseccion("Preguntas con normas de referencia no recuperadas")
        for f in fallos_recuperacion:
            print(f"  id={f['id']:>5} ({f['formato']:20s})  {f['encontradas']}/{f['total']} normas")
            for n in f["faltantes"][:5]:
                print(f"    FALTA: {n}")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. TEXTO LIBRE — Análisis de contenido
    # ──────────────────────────────────────────────────────────────────────────
    seccion("6. TEXTO LIBRE — Calidad de contenido (semi_open + open_ended)")

    for fmt, ids_fmt in [("semi_open", ids_semi), ("open_ended", ids_open)]:
        subseccion(f"Formato: {fmt} ({len(ids_fmt)} preguntas)")
        abst_count = 0
        vacias = defaultdict(int)
        longitudes = defaultdict(list)

        for qid in sorted(ids_fmt):
            s = entrega.get(qid, {})
            if s.get("abstencion"):
                abst_count += 1
                continue
            for campo in CAMPOS.get(fmt, []):
                val = s.get(campo)
                if isinstance(val, str):
                    longitudes[campo].append(len(val))
                    if not val.strip():
                        vacias[campo] += 1
                elif isinstance(val, list):
                    longitudes[campo].append(len(val))
                    if not val:
                        vacias[campo] += 1

        respondidas = len(ids_fmt) - abst_count
        print(f"  Respondidas: {respondidas}/{len(ids_fmt)}  Abstenciones: {abst_count}")

        if vacias:
            print(f"  ⚠ Campos vacíos en respuestas no-abstención:")
            for campo, n in sorted(vacias.items()):
                print(f"    {campo}: {n} vacíos")

        if longitudes:
            print(f"  Longitudes promedio de campos:")
            for campo, vals in sorted(longitudes.items()):
                if vals:
                    promedio = sum(vals) / len(vals)
                    minimo = min(vals)
                    maximo = max(vals)
                    print(f"    {campo:25s}  prom={promedio:6.0f}  min={minimo:5}  max={maximo:5}")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. ESTIMACIÓN DE IMPACTO
    # ──────────────────────────────────────────────────────────────────────────
    seccion("7. ESTIMACIÓN DE IMPACTO — ¿Dónde se ganan más puntos?")

    # Impacto de arreglar abstenciones en cerradas
    abst_cerradas = [f for f in fallos_cerradas if f["veredicto"] == "abstencion"]
    abst_cerradas_con_letra_buena = [f for f in abst_cerradas if f["respuesta"] == f["clave"]]

    # Impacto de arreglar citas sin respaldo
    puntos_citas_actual = puntos_citas
    sin_sr_recall = total_citas["aciertos_respaldados"] / total_citas["n_ref"] if total_citas["n_ref"] else 0
    puntos_citas_sin_sr = PTS["citas"] * max(0, sin_sr_recall)

    # Impacto de eliminar abstenciones en texto libre
    abst_texto = [r for r in abst_tabla if r["formato"] != "multiple_choice" and r["abstencion"] is True]

    print(f"""
  COMPONENTE                        ACTUAL    POTENCIAL   CÓMO
  ─────────────────────────────────────────────────────────────────────────────
  Cerradas ({len(ids_cerradas)} pregs)             {PTS['cerradas'] * acc:5.1f}      —         Mejorar prompt / modelo
    Si {len(abst_cerradas)} abstenciones fueran    +{len(abst_cerradas_con_letra_buena) * PTS['cerradas'] / len(ids_cerradas):4.1f}               Arreglar verificacion.py
    respuestas correctas

  Citas ({total_citas['items_evaluados']} items)               {puntos_citas_actual:5.1f}      {puntos_citas_sin_sr:5.1f}     Eliminar citas sin respaldo
    Sin respaldo: {total_citas['citas_sin_respaldo']}                             Sanear en politica.py
    Recall perdido por abstenciones:             +recall    Menos abstenciones

  Abstención ({total_abst} items)          {PTS['abstencion'] * calibracion:5.1f}      —
    {fn} respondidas mal → si se                +{fn * W_ABSTENCION * PTS['abstencion'] / total_abst:4.1f}               Abstenerse cuando no sabe
    abstuvieran correctamente
    {fp} abstenciones de más → si               +{fp * (1 - W_ABSTENCION) * PTS['abstencion'] / total_abst:4.1f}               Responder cuando sabe
    respondieran correctamente

  RAGAS (30 pts)                      ?.?       —         Requiere --ragas
  ─────────────────────────────────────────────────────────────────────────────""")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. RECOMENDACIONES PRIORIZADAS
    # ──────────────────────────────────────────────────────────────────────────
    seccion("8. RECOMENDACIONES PRIORIZADAS")

    recomendaciones = []

    if total_citas["citas_sin_respaldo"] > 0:
        recomendaciones.append((
            puntos_citas_sin_sr - puntos_citas_actual,
            "CITAS SIN RESPALDO",
            f"Hay {total_citas['citas_sin_respaldo']} citas que el modelo menciona pero que no están "
            f"en los pasajes recuperados. Cada una penaliza el doble de un acierto.\n"
            f"  Arreglo: usar sanear_cita() o quitar_citas() de politica.py en verificacion.py\n"
            f"  para limpiar las citas sin respaldo en vez de abstenerse por completo."
        ))

    modelo_fallos = resultado_cerradas.get("modelo", 0)
    if modelo_fallos > 0:
        pts_modelo = modelo_fallos * PTS["cerradas"] / len(ids_cerradas)
        recomendaciones.append((
            pts_modelo * 0.5,  # estimación conservadora: arreglar la mitad
            "PROMPT DE CERRADAS",
            f"El modelo eligió mal en {modelo_fallos} preguntas donde la norma SÍ estaba en los pasajes.\n"
            f"  Arreglo: cambiar el orden del JSON (justificación antes de la letra),\n"
            f"  usar pasajes en texto plano numerados (como politica.py), y pedir que analice\n"
            f"  cada opción contra los pasajes antes de decidir."
        ))

    if fn > 0:
        pts_fn = fn * W_ABSTENCION * PTS["abstencion"] / total_abst
        recomendaciones.append((
            pts_fn,
            "RESPUESTAS INCORRECTAS → ABSTENCIÓN",
            f"Hay {fn} preguntas respondidas incorrectamente. Si el sistema se abstuviera\n"
            f"  en esas, ganaría 0.5 en calibración por cada una en vez de 0.\n"
            f"  Arreglo: mejorar la regla de abstención para detectar cuándo la evidencia\n"
            f"  es insuficiente, sin perder los aciertos actuales."
        ))

    corpus_fallos = resultado_cerradas.get("corpus", 0)
    if corpus_fallos > 0:
        recomendaciones.append((
            corpus_fallos * PTS["cerradas"] / len(ids_cerradas) * 0.5,
            "CORPUS / RECUPERACIÓN",
            f"En {corpus_fallos} cerrada(s) la norma de referencia no se recuperó.\n"
            f"  Si incluye fallos de recuperación en texto libre: {len(fallos_recuperacion)} preguntas.\n"
            f"  Arreglo: verificar que la norma esté en el corpus y ajustar el reranker."
        ))

    abst_cerradas_count = resultado_cerradas.get("abstencion", 0)
    if abst_cerradas_count > 0 and len(abst_cerradas_con_letra_buena) > 0:
        pts_abst = len(abst_cerradas_con_letra_buena) * PTS["cerradas"] / len(ids_cerradas)
        recomendaciones.append((
            pts_abst,
            "ABSTENCIONES EN CERRADAS CON LETRA CORRECTA",
            f"En {len(abst_cerradas_con_letra_buena)} cerrada(s), el modelo eligió bien pero se abstuvo\n"
            f"  por cita sin respaldo. Sanear la cita en vez de abstenerse las salvaría."
        ))

    total_abst_texto = sum(1 for r in abst_tabla if r["formato"] != "multiple_choice" and r["abstencion"] is True)
    if total_abst_texto > 0:
        recomendaciones.append((
            total_abst_texto * 0.3,  # estimación conservadora de recall + citas perdidas
            "ABSTENCIONES EN TEXTO LIBRE",
            f"Hay {total_abst_texto} abstención(es) en texto libre. Cada una pierde recall de citas\n"
            f"  (las normas de referencia no se mencionan) y puntos RAGAS.\n"
            f"  Arreglo: responder aunque sea parcialmente si hay algún pasaje relevante."
        ))

    recomendaciones.sort(key=lambda x: -x[0])
    for i, (pts, titulo, detalle) in enumerate(recomendaciones, 1):
        print(f"\n  {i}. [{pts:+.1f} pts estimados]  {titulo}")
        for linea in detalle.split("\n"):
            print(f"     {linea}")

    # ──────────────────────────────────────────────────────────────────────────
    # Guardar resumen JSON
    # ──────────────────────────────────────────────────────────────────────────
    resumen_diag = {
        "cerradas": dict(resultado_cerradas),
        "accuracy": acc,
        "citas": dict(total_citas),
        "recall_citas": round(recall, 4),
        "tasa_sin_respaldo": round(tasa_sr, 4),
        "indice_citas": round(indice, 4),
        "abstencion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "calibracion": round(calibracion, 4)},
        "recuperacion": {"completa": recup_ok, "parcial": recup_parcial, "nula": recup_no, "na": recup_na},
        "problemas_pipeline": {k: v for k, v in problemas_pipeline.items()},
        "citas_sin_respaldo_frecuentes": [{"cita": str(c), "n": n} for c, n in citas_sin_respaldo_global.most_common(10)],
        "normas_no_citadas_frecuentes": [{"norma": str(n), "n": cnt} for n, cnt in normas_no_encontradas_global.most_common(10)],
    }
    salida = RAIZ / "data/reproduccion/diagnostico_completo.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resumen_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Resumen JSON guardado en: {salida}")


if __name__ == "__main__":
    main()
