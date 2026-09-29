"""Genera el notebook auditable desde celdas fuente versionadas."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks/experimentos/e06_corpus_definitivo.ipynb"


def md(value):
    return nbf.v4.new_markdown_cell(value)


def code(value):
    return nbf.v4.new_code_cell(value)


cells = [
    md("""# E06 · Corpus definitivo: recuperación y respuesta jurídica

**Objetivo:** comparar variantes secuenciales sobre `corpus_eval_v1` (13.962 documentos), en RTX 4090, A100 o L4. Los 50 ítems públicos se usan como evaluación; sólo `id`, pregunta, formato, opciones y metadatos permitidos llegan al sistema. `legal_basis` y respuestas nunca entran en el índice, consulta o prompt.

Los resultados reales se escriben en `data/experimentos/corpus_definitivo/`. Una ejecución completa descarga varios modelos y crea índices densos de todo el corpus; puede tardar horas o días. El kernel debe permanecer en un entorno con CUDA, PyTorch y `requirements-notebook-definitivo.txt`. La [guía](../../GUIA_NOTEBOOK_DEFINITIVO.md) tiene los comandos para Windows y Colab. Ejecuta las celdas en orden; si una etapa se interrumpe, reanuda desde ella."""),
    md("## 0. Directorio, semilla y política de ejecución\n`ROOT` puede declararse como variable de entorno `AI_WEEK_ROOT`. En Colab, extrae primero el paquete portable en `/content/ai-week-2026` y abre este notebook."),
    code("""import os, sys, json, random
from pathlib import Path

def locate_root():
    choices = [Path(os.environ['AI_WEEK_ROOT'])] if os.environ.get('AI_WEEK_ROOT') else []
    choices += [Path.cwd(), *Path.cwd().parents, Path('/content/ai-week-2026')]
    for p in choices:
        if (p / 'data/releases/corpus_eval_v1/snapshot.json').is_file():
            return p.resolve()
    raise FileNotFoundError('No encuentro el proyecto. Define AI_WEEK_ROOT o extrae el paquete de Colab.')

ROOT = locate_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from scripts.experimentos import corpus_definitivo as exp

# El barrido completo es la ejecución para decidir. Para validar instalación primero,
# ejecuta sólo los tests del repositorio; no selecciones modelos con un piloto parcial.
RUN_4B_ENCODER = exp.hardware()['ram_gib'] >= 32
if not RUN_4B_ENCODER:
    print('Qwen3-Embedding-4B omitido: FAISS Flat 2560D requiere más RAM; se documenta como no ejecutado.')
RUN_QWEN_RERANKER = True
RUN_DECODERS = True
RUN_RAGAS = False  # Activar sólo con la clave oficial y dependencias del juez.
print('Proyecto:', ROOT)
print('Salida:', exp.RUNS)"""),
    md("## 1. Preflight: snapshot, licencia/parámetros, CUDA y entradas sin respuestas\nVerifica los hashes de los 13.962 textos. Ésta es una operación de lectura y puede tardar varios minutos."),
    code("""preflight = exp.preflight(verify_hashes=True)
display(preflight)
items = exp.questions()
assert preflight['snapshot_id'] == '7eabf300335b23be5541540c34b1828c15c0a05b3f65c595a22ff8605b8ebd77'
assert preflight['documentos'] == 13962 and preflight['preguntas'] == 50
assert preflight['textos_sha256_verificados'] is True
assert preflight['hardware']['vram_gib'] >= 20
assert {q['formato'] for q in items} <= {'multiple_choice', 'semi_open', 'open_ended'}
assert all('legal_basis' not in x and 'respuesta_correcta' not in x for x in items)
print('Formatos:', {f: sum(q['formato']==f for q in items) for f in sorted({q['formato'] for q in items})})"""),
    md("## 2. Construir unidades literales y BM25\nSegmenta normas por artículos cuando la atribución es segura y conserva también providencias, conceptos y compendios. Los pasajes finales son cortes exactos de los archivos del snapshot, con offsets. Un proceso interrumpido durante esta celda reconstruye SQLite desde cero; las siguientes etapas sí reanudan por checkpoint."),
    code("""lexical = exp.build_lexical()
display(lexical)
assert not lexical['piloto'] and lexical['documentos'] == 13962

if not (exp.RUNS/'rankings/R00.json').exists():
    r00 = {str(q['id']): exp.bm25(q, 100) for q in items}
    exp.save_ranking('R00', r00)
else:
    r00 = exp.load_ranking('R00')
display(exp.evaluate_retrieval('R00', r00))"""),
    md("## 3. BGE-M3: denso → híbrido RRF → reranker BGE\nLos tres resultados comparten índice y preguntas. No se generan respuestas aún; así aislamos la mejora del recuperador."),
    code("""exp.build_dense('BAAI/bge-m3')
if not (exp.RUNS/'rankings/R01.json').exists():
    r01 = exp.dense_results('BAAI/bge-m3', items)
    exp.save_ranking('R01', r01)
else:
    r01 = exp.load_ranking('R01')
display(exp.evaluate_retrieval('R01', r01))

if not (exp.RUNS/'rankings/R02.json').exists():
    r02 = {str(q['id']): exp.rrf(r00[str(q['id'])], r01[str(q['id'])]) for q in items}
    exp.save_ranking('R02', r02)
else:
    r02 = exp.load_ranking('R02')
display(exp.evaluate_retrieval('R02', r02))

if not (exp.RUNS/'rankings/R03.json').exists():
    r03 = exp.rerank_many(items, r02)
    exp.save_ranking('R03', r03)
else:
    r03 = exp.load_ranking('R03')
display(exp.evaluate_retrieval('R03', r03))"""),
    md("## 4. E5 multilingual, Qwen3 Embedding 0.6B y 4B\nCada encoder construye su propio índice de **todos** los pasajes. El reranker BGE es el mismo para que la comparación sea controlada. La opción 4B puede consumir mucho espacio y tiempo; si se desactiva, queda ausente de la comparación y no se interpreta como perdedora."),
    code("""challengers = [
    ('R04', 'intfloat/multilingual-e5-large'),
    ('R05', 'Qwen/Qwen3-Embedding-0.6B'),
]
if RUN_4B_ENCODER:
    challengers.append(('R06', 'Qwen/Qwen3-Embedding-4B'))
for run_id, encoder_name in challengers:
    exp.build_dense(encoder_name)
    if not (exp.RUNS/f'rankings/{run_id}_hybrid.json').exists():
        dense = exp.dense_results(encoder_name, items)
        hybrid = {str(q['id']): exp.rrf(r00[str(q['id'])], dense[str(q['id'])]) for q in items}
        exp.save_ranking(run_id+'_hybrid', hybrid)
    else:
        hybrid = exp.load_ranking(run_id+'_hybrid')
    display(exp.evaluate_retrieval(run_id+'_hybrid', hybrid))
    if not (exp.RUNS/f'rankings/{run_id}.json').exists():
        reranked = exp.rerank_many(items, hybrid)
        exp.save_ranking(run_id, reranked)
    else:
        reranked = exp.load_ranking(run_id)
    display(exp.evaluate_retrieval(run_id, reranked))"""),
    md("## 5. Retador de reranking Qwen3 0.6B\nReordena **el mismo top 50 híbrido BGE** de R03; sólo cambia el reranker. Es una ablación independiente."),
    code("""if RUN_QWEN_RERANKER:
    if not (exp.RUNS/'rankings/R07.json').exists():
        r07 = exp.rerank_qwen_many(items, r02)
        exp.save_ranking('R07', r07)
    else:
        r07 = exp.load_ranking('R07')
    display(exp.evaluate_retrieval('R07', r07))"""),
    md("## 6. Decidir dos recuperadores antes de cargar decoders\nLa métrica de citas es un **proxy**, no el puntaje oficial ni recall documental: el paquete de 50 no publica doc_id relevante. La selección se hace sólo con 35 preguntas de desarrollo fijadas por id; se presentan luego 15 restantes por separado. Guarda la decisión antes de generar."),
    code("""import hashlib, pandas as pd
development = [q['id'] for q in items if int(hashlib.sha256(str(q['id']).encode()).hexdigest(), 16) % 10 < 7]
holdout = [q['id'] for q in items if q['id'] not in development]
candidates = ['R00', 'R02', 'R03', 'R04', 'R05'] + (['R06'] if RUN_4B_ENCODER else []) + (['R07'] if RUN_QWEN_RERANKER else [])
dev = [exp.evaluate_retrieval(name+'_dev', exp.load_ranking(name), development) for name in candidates]
hold = [exp.evaluate_retrieval(name+'_holdout', exp.load_ranking(name), holdout) for name in candidates]
display(pd.DataFrame(dev).sort_values('respaldo_literal_top10', ascending=False))
display(pd.DataFrame(hold).sort_values('respaldo_literal_top10', ascending=False))
selection = sorted(dev, key=lambda r: ((r['respaldo_literal_top10'] or 0), (r['MRR_cita_top10'] or 0)), reverse=True)[:2]
chosen = [x['experimento'].replace('_dev','') for x in selection]
exp.dump(exp.RUNS/'seleccion_recuperadores.json', {'desarrollo_ids': development,
          'holdout_ids': holdout, 'seleccion': chosen, 'criterio': 'proxy de citas top10 en desarrollo'})
print('Recuperadores seleccionados:', chosen)"""),
    md("## 7. Decoders con evidencia fijada, JSON y evaluador oficial\nSe ejecutan hasta seis combinaciones (dos recuperadores × tres decoders), secuencialmente. Temperatura cero equivale a `do_sample=False`. Cada ítem se guarda inmediatamente para reanudar. Si el JSON es inválido o una cita identificada carece de respaldo, se registra abstención. El juez RAGAS exige la clave oficial; sin ella el informe refleja sólo **50 puntos deterministas de 80 automáticos**, no un total comparable a 100."),
    code("""scored = []
if RUN_DECODERS:
    for retriever in chosen:
        for decoder in exp.DECODERS:
            submission = exp.generate(decoder, retriever, items)
            score = exp.official_score(submission, ragas=RUN_RAGAS)
            assert 0 <= score['cerradas']['puntos'] <= 20
            assert 0 <= score['citas']['puntos'] <= 20
            assert 0 <= score['abstencion']['puntos'] <= 10
            assert score['total_automatico']['posibles'] == (80 if RUN_RAGAS else 50)
            outputs = exp.read_jsonl(submission)
            details = exp.read_jsonl(submission.parent/'detalles.jsonl')
            timings = [x['latencia_generacion_ms'] for x in details]
            resources = json.loads((submission.parent/'recursos.json').read_text(encoding='utf-8'))
            scored.append({'recuperador': retriever, 'decoder': decoder,
                           'cerradas': score['cerradas']['puntos'],
                           'citas': score['citas']['puntos'],
                           'abstencion': score['abstencion']['puntos'],
                           'ragas': score['correccion_ragas']['puntos'],
                           'determinista_50': sum(score[k]['puntos'] for k in ('cerradas','citas','abstencion')),
                           'automatico_obtenido': score['total_automatico']['obtenidos'],
                           'automatico_posible': score['total_automatico']['posibles'],
                           'errores_formato': score['validacion']['errores'],
                           'abstenciones': sum(x['abstencion'] for x in outputs),
                           'generacion_p50_ms': float(np.percentile(timings, 50)),
                           'generacion_p95_ms': float(np.percentile(timings, 95)),
                           'vram_pico_gib': resources['vram_pico_gib'],
                           'ruta_entrega': str(submission)})
            display(scored[-1])
    exp.dump(exp.RUNS/'comparacion_final.json', scored)
display(pd.DataFrame(scored).sort_values('determinista_50', ascending=False) if scored else 'Generación desactivada')"""),
    md("## 8. Lectura de resultados y próxima decisión\nCompara precisión cerrada, citas con evidencia, abstención, errores de formato y latencia. `determinista_50` no es la nota final: faltan 30 de RAGAS y 20 de interfaz, bitácora, vídeo y reproducibilidad. Si el proxy mejora pero citas oficiales no, inspecciona la evidencia literal y la recuperación del artículo completo. Si suben citas pero cae la exactitud cerrada, mantén evidencia fija y ajusta el decoder/prompts. Si las abstenciones dominan, revisa errores de formato y límites de contexto antes de ajustar umbrales."),
    code("""display(pd.DataFrame(exp.compare()).sort_values('respaldo_literal_top10', ascending=False))
if (exp.RUNS/'comparacion_final.json').exists():
    final = pd.DataFrame(json.loads((exp.RUNS/'comparacion_final.json').read_text(encoding='utf-8')))
    display(final.sort_values(['errores_formato','automatico_obtenido'], ascending=[True,False]))
print('Artefactos:', exp.RUNS)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (CUDA)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
nbf.validate(nb)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
