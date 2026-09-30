# Prueba de fuerza en la RTX 4090 (rama `testing`)

Todo el recorrido sobre el corpus completo y las 50 preguntas de muestra, desde un solo archivo.

```bash
# 0. Código
git clone https://github.com/SamuelCharry/ai-week-2026.git && cd ai-week-2026   # o: git fetch
git checkout testing
# Copiar data/raw (no está en git) dentro de ai-week-2026/data/raw

# 1. Gráfica: debe listar la RTX 4090 y un driver compatible con CUDA 12.8 (570 o posterior)
nvidia-smi

# 2. Entorno
python3 -m venv .venv-sistema && source .venv-sistema/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r data/oficial/scripts/requirements-evaluador.txt
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30,1), 'GiB')"

# 3. Todo: textos, inventario, BM25, índice BGE-M3, respuestas y evaluador oficial
python3 src/main.py --desde-raw
```

La primera vez prepara ~14.000 documentos (CPU), calcula el índice BGE-M3 de ~1,3 millones de fragmentos
(GPU, horas) y descarga ~20 GB de modelos. Si se corta, el mismo comando continúa donde iba.
Resultados: `data/reproduccion/submissions_sample_reporte.json` (puntaje) y
`data/reproduccion/submissions_sample_resumen.json` (segundos por pregunta frente a los 22 s del sábado).
Cada respuesta queda con su texto crudo, sus probabilidades de letra y lo que se corrigió en
`data/reproduccion/submissions_sample_respuestas/<id>.json`.

Con el índice ya construido, comparar encoders y decoders antes de decidir:

```bash
python3 src/comparar.py --recuperacion
python3 src/comparar.py --sistema --variantes qwen25-7b qwen3-4b-2507
```

La prueba en miniatura sin GPU (`python3 src/prueba_local.py --conjunto amplio`) dio 41,33/50 en 19
preguntas con un decoder de 0,5B; en la 4090 corre el Qwen2.5-7B.

# Cómo correr el sistema (opción A) en una RTX 4090

BM25 + BGE-M3 con RRF, reranker BGE-v2-m3 y Qwen2.5-7B-Instruct. Todo se ejecuta desde la raíz
del repositorio. Los comandos están para Linux; al final hay equivalentes para Windows.

## Atajo: un solo archivo

Después de instalar el entorno (paso 1), `src/main.py` hace los pasos 2 a 4: encuentra los datos
dentro de la carpeta donde se descomprimieron, los enlaza, construye el índice si falta, responde y
evalúa.

```bash
python3 src/main.py --datos datos --solo-preparar   # revisa entorno, datos e índice
python3 src/main.py --datos datos --prueba          # 3 preguntas
python3 src/main.py --datos datos                   # las 50 de muestra + evaluador oficial
python3 src/main.py --datos datos --split test      # las 992 del sábado
```

### Desde `data/raw` (sin el paquete del Drive)

Si solo están los originales en `data/raw`, `--desde-raw` prepara los textos, arma el inventario
`data/releases/corpus_eval_v1` y construye el índice. La preparación y el BM25 no necesitan GPU:

```bash
python3 src/main.py --desde-raw --solo-corpus    # sin GPU: textos canónicos, inventario y BM25
python3 src/main.py --desde-raw --prueba         # con GPU: índice BGE-M3 (si falta) y 3 respuestas
python3 src/main.py --desde-raw                  # las 50 de muestra + evaluador oficial
```

Si la preparación se hace en otro equipo, copiar a la máquina con GPU `data/processed/corpus_preparado/`,
`data/releases/corpus_eval_v1/` y `data/experimentos/corpus_definitivo/`.

### Comparar variantes antes de decidir (`src/comparar.py`)

Con el índice listo, mide en la misma corrida las alternativas de encoder, recuperación y decoder
sobre las 50 preguntas de muestra, y deja una tabla en `data/comparacion/`:

```bash
python3 src/comparar.py --recuperacion                               # minutos, sin decoder
python3 src/comparar.py --sistema --variantes qwen25-7b qwen3-4b-2507  # con el evaluador oficial
```

Otros encoders (E5, Qwen3-Embedding) solo se miden si E06 dejó su índice en
`data/experimentos/corpus_definitivo/indices/`.

## 0. Requisitos

| Recurso | Mínimo | Por qué |
|---|---|---|
| GPU | 24 GB de VRAM (RTX 4090) | Qwen2.5-7B en bfloat16 ocupa ~15 GB; encoder y reranker, ~1 GB cada uno |
| Driver NVIDIA | compatible con CUDA 12.8 (serie 570 o posterior) | PyTorch se instala con ruedas cu128 |
| RAM | 32 GB | El índice FAISS (1.304.984 × 1.024 float32) ocupa ~5 GB en memoria |
| Disco | ~40 GB libres | Pesos de los tres modelos (~20 GB), índice, `chunks.sqlite` y textos |
| Python | 3.12 | |

## 1. Código y entorno

```bash
git pull
python3.12 -m venv .venv-sistema
source .venv-sistema/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r data/oficial/scripts/requirements-evaluador.txt
pip install -e .
```

Comprobar:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m legalrag --help
```

La primera línea debe imprimir `True NVIDIA GeForce RTX 4090`.

## 2. Datos: corpus, manifiesto e índice

El sistema necesita estos archivos (rutas en `configs/sistema.json`, sección `recuperacion`):

| Archivo | De dónde sale |
|---|---|
| `data/processed/corpus_preparado/textos/*.txt` | Paquete `ai-week-2026-corpus-definitivo.zip` (Drive, carpeta AIWEEK) |
| `data/releases/corpus_eval_v1/corpus_manifest.json` | Mismo paquete |
| `data/experimentos/corpus_definitivo/chunks.sqlite` | Resultados de E06 (paso 2.2) |
| `data/experimentos/corpus_definitivo/indices/BAAI__bge-m3/index.faiss` y `complete.json` | Resultados de E06 (paso 2.2) |

### 2.1 Extraer el paquete del corpus

Descargar `ai-week-2026-corpus-definitivo.zip` a la raíz del repositorio y extraer solo los datos:

```bash
python - <<'EOF'
import zipfile
from pathlib import Path
with zipfile.ZipFile("ai-week-2026-corpus-definitivo.zip") as z:
    for nombre in z.namelist():
        relativo = nombre.split("/", 1)[1] if "/" in nombre else ""
        if relativo.startswith(("data/processed/", "data/releases/", "reports/")) and not nombre.endswith("/"):
            destino = Path(relativo)
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(z.read(nombre))
print("listo")
EOF
```

No se extrae el código viejo que trae el paquete (`scripts/…`): el repositorio ya tiene la versión actual.

### 2.2 Índice: copiar el de E06 o reconstruirlo

**Opción recomendada — copiar el de E06.** La guía de E06 respalda los resultados en
`MyDrive/AIWEEK/e06-resultados/`. Copiar esa carpeta completa como
`data/experimentos/corpus_definitivo/`. Debe contener `chunks.sqlite` e
`indices/BAAI__bge-m3/` con `index.faiss` y `complete.json`.

**Alternativa — reconstruirlo en la 4090.** Usa el mismo código de E06:

```bash
python -c "from legalrag.experimentos import corpus_definitivo as e; print(e.build_lexical()); print(e.build_dense('BAAI/bge-m3'))"
```

`build_lexical` segmenta los 13.962 documentos (unos 5 minutos en E06). `build_dense` codifica
1,3 millones de fragmentos y guarda puntos de control: si se interrumpe, volver a correr el mismo
comando continúa desde el último. Como segmenta con el `ingesta.py` actual del repositorio, los ids de
fragmento pueden no coincidir con los rankings guardados de E06; el sistema queda consistente consigo
mismo.

### 2.3 Comprobar

```bash
python - <<'EOF'
import json, sqlite3
from pathlib import Path
import faiss
from legalrag.config import RAIZ, leer_config
c = leer_config()["recuperacion"]
for clave in ("fragmentos", "indice_denso", "indice_meta", "textos", "manifiesto"):
    print(clave, (RAIZ / c[clave]).exists())
total = sqlite3.connect(RAIZ / c["fragmentos"]).execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
indice = faiss.read_index(str(RAIZ / c["indice_denso"]))
meta = json.loads((RAIZ / c["indice_meta"]).read_text())
print("fragmentos", total, "vectores", indice.ntotal, "encoder", meta["modelo"], meta["revision"][:8])
EOF
```

Todo debe decir `True`, fragmentos y vectores deben coincidir, y el encoder debe ser `BAAI/bge-m3 5617a9f6`.

## 3. Prueba corta: tres preguntas

```bash
python -m legalrag answer --split sample --ids 51 79 140 --salida data/reproduccion/prueba.jsonl
```

La primera vez descarga Qwen2.5-7B, BGE-M3 y el reranker (~20 GB). Con muchas descargas conviene
exportar `HF_TOKEN`. Al terminar, revisar:

- `data/reproduccion/prueba_resumen.json`: `respondidas`, `abstenciones`, `errores`, `errores_esquema`
  y `segundos_promedio`.
- `data/reproduccion/prueba_respuestas/<id>.json`: la respuesta, los segundos y `problema`, que dice
  por qué se abstuvo (`cita_sin_respaldo_o_indeterminada`, `json_invalido:…`, `abstencion_del_modelo`,
  `sin_evidencia_en_contexto`) o `null` si respondió.

Si `segundos_promedio` pasa de 22, ver la sección 8 antes de correr las 50.

## 4. Las 50 preguntas de muestra y el evaluador oficial

```bash
python -m legalrag answer --split sample
python -m legalrag evaluate --submission data/reproduccion/submissions_sample.jsonl --split sample \
    --out data/reproduccion/reporte_sample.json
```

`answer` guarda cada respuesta al terminarla: si se corta, el mismo comando continúa donde iba.
`--sin-reanudar` fuerza a responder todo de nuevo. El evaluador sin `--ragas` da los 50 puntos
deterministas (cerradas 20, citas 20, abstención 10). Para los 30 de texto libre:

```bash
export OPENROUTER_API_KEY=...
python -m legalrag evaluate --submission data/reproduccion/submissions_sample.jsonl --split sample --ragas \
    --out data/reproduccion/reporte_sample_ragas.json
```

La llave solo se usa en la evaluación; el sistema no la lee.

## 5. Determinismo (verificación en vivo)

El jurado regenera dos o tres respuestas y las compara con las entregadas. Probarlo:

```bash
python -m legalrag answer --split sample --ids 51 79 --sin-reanudar --salida data/reproduccion/repeticion.jsonl
python - <<'EOF'
import json
a = {r["id"]: r for r in map(json.loads, open("data/reproduccion/submissions_sample.jsonl"))}
b = {r["id"]: r for r in map(json.loads, open("data/reproduccion/repeticion.jsonl"))}
for i in b:
    print(i, "idénticas" if a[i] == b[i] else "DISTINTAS")
EOF
```

## 6. Interfaz

En dos terminales:

```bash
python -m legalrag serve --puerto 8000
python -m http.server 8766 --directory interfaz --bind 127.0.0.1
```

Abrir http://127.0.0.1:8766/. El servicio carga los modelos una vez y atiende una consulta a la vez.

## 7. Sábado: las 992 preguntas

1. Copiar `test_992.jsonl` a `data/oficial/data/`.
2. `python -m legalrag answer --split test` escribe `submissions.jsonl` en la raíz. Si se corta,
   repetir el mismo comando.
3. Revisar `submissions_resumen.json`: `respondidas` debe ser 992 y `errores_esquema` una lista vacía.
4. No cambiar `configs/sistema.json` ni el índice después de la entrega.

## 8. Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| `CUDA out of memory` al reordenar | Bajar `lote_reranker` de 6 a 3 en `configs/sistema.json` |
| `CUDA out of memory` al generar | Bajar `contexto` a 4096 |
| Más de 22 s por pregunta | Bajar `rerank_top` de 50 a 30 o `max_nuevos_tokens` de 600 a 450, y medir de nuevo |
| `El índice denso se construyó con otro encoder o revisión` | El `complete.json` no es de BGE-M3 5617a9f6: recuperar el índice correcto |
| `El índice denso y chunks.sqlite no tienen los mismos fragmentos` | `chunks.sqlite` e índice son de corridas distintas: copiar ambos de la misma carpeta |
| Muchas abstenciones `cita_sin_respaldo_o_indeterminada` | Revisar las respuestas: el modelo cita normas que no están en los pasajes |
| Muchas `json_invalido` | Revisar el texto crudo aumentando `max_nuevos_tokens`; puede estar cortándose |

Cambiar un parámetro invalida las respuestas guardadas (la firma incluye la configuración), así que
la siguiente corrida responde todo de nuevo. Registrar cada medición en `CORPUS.md`, sección 4.

## Windows (PowerShell)

```powershell
py -3.12 -m venv .venv-sistema
.\.venv-sistema\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r data\oficial\scripts\requirements-evaluador.txt
pip install -e .
python -m legalrag answer --split sample --ids 51 79 140 --salida data\reproduccion\prueba.jsonl
```

Los bloques `python - <<'EOF'` se guardan en un archivo `.py` y se ejecutan con `python archivo.py`.
