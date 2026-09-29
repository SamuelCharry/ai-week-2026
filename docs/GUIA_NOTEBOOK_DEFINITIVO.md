# Ejecutar E06 en RTX 4090, A100 o L4

El notebook [E06](../notebooks/experimentos/e06_corpus_definitivo.ipynb) compara BM25, cuatro encoders, dos rerankers y tres decoders sobre el snapshot `corpus_eval_v1`. Usa exclusivamente las preguntas públicas de `data/oficial/data/sample_50.jsonl`. El módulo [corpus_definitivo.py](../src/legalrag/experimentos/corpus_definitivo.py) guarda el índice y los resultados en `data/experimentos/corpus_definitivo/`; no modifica el corpus. El archivo de muestra **sí** trae respuestas y `legal_basis`: el pipeline los elimina antes de recuperar o generar, y sólo el evaluador oficial los consulta al puntuar.

## Windows, RTX 4090

Desde PowerShell en la raíz del proyecto:

```powershell
py -3.12 -m venv .venv-e06
.\.venv-e06\Scripts\python.exe -m pip install --upgrade pip
.\.venv-e06\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv-e06\Scripts\python.exe -m pip install -r requirements/notebook-definitivo.txt
.\.venv-e06\Scripts\python.exe -m ipykernel install --user --name ai-week-e06 --display-name "AI Week E06 CUDA"
.\.venv-e06\Scripts\python.exe -m jupyter lab
```

Abre `notebooks/experimentos/e06_corpus_definitivo.ipynb`, elige el kernel **AI Week E06 CUDA** y ejecuta todas las celdas en orden. La primera celda exige CUDA y muestra GPU/VRAM/RAM. Para ahorrar disco, el notebook no copia `data/raw` ni vuelve a perfilarlo. Si el controlador no admite CUDA 12.8, selecciona en [PyTorch Start Locally](https://pytorch.org/get-started/locally/) la variante CUDA compatible con tu driver.

El índice denso completo puede ocupar varios GiB por encoder y descargar otros tantos de pesos; conviene disponer de decenas de GiB libres. Con 15 GiB de RAM en la máquina local, cierra aplicaciones pesadas antes de ejecutar las etapas FAISS. El encoder Qwen3-4B (2560 dimensiones, índice Flat) se omite automáticamente si la RAM es menor de 32 GiB; esa fila no puede considerarse evaluada. La carga es secuencial y el lote se reduce automáticamente si la GPU se queda sin VRAM. Los índices de cada encoder guardan checkpoints; al reconectar, ejecuta desde la etapa pendiente. El índice léxico se reconstruye si se interrumpe antes del marcador de finalización.

## Google Colab, A100 o L4

El paquete privado contiene los 13.962 textos seleccionados y los archivos necesarios, **sin** `data/raw`, respuestas adicionales ni pesos descargados. Créalo una vez en Windows:

```powershell
.\.venv\Scripts\python.exe scripts\experimentos\empaquetar_colab_definitivo.py
```

Sube `data/colab/ai-week-2026-corpus-definitivo.zip` a tu Drive privado. En Colab selecciona **Entorno de ejecución → Cambiar tipo → GPU A100 o L4**. Luego, en una celda de preparación:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import zipfile
paquete = Path('/content/drive/MyDrive/ai-week-2026-corpus-definitivo.zip')
with zipfile.ZipFile(paquete) as z:
    z.extractall('/content')
%pip install -r /content/ai-week-2026/requirements/notebook-definitivo.txt
import os
os.environ['AI_WEEK_ROOT'] = '/content/ai-week-2026'
```

Reinicia el entorno de ejecución tras instalar dependencias si Colab lo pide; tras reiniciar vuelve a montar Drive y definir `AI_WEEK_ROOT`. Abre o sube el E06 extraído y ejecuta desde la primera celda. Los resultados en `/content` desaparecen al terminar la sesión. Al final de **cada etapa densa o generativa**, guarda el estado:

```python
from pathlib import Path
import shutil
origen = Path('/content/ai-week-2026/data/experimentos/corpus_definitivo')
respaldo = Path('/content/drive/MyDrive/AIWEEK/e06-resultados')
shutil.copytree(origen, respaldo, dirs_exist_ok=True)
```

Después de reconectar y extraer el ZIP, restaura el estado antes de correr el notebook:

```python
shutil.copytree(respaldo, origen, dirs_exist_ok=True)
```

Copiar los índices de varios GiB puede tardar; espera a que termine. No ejecutes a la vez la misma carpeta de resultados desde dos runtimes; cada índice se construye con la revisión de modelo fijada en el catálogo.

La A100 permite lotes y contexto inicial mayor; L4 y RTX 4090 usan lotes más pequeños. La configuración se deriva de la VRAM real y retrocede ante OOM. No se afirma que las latencias o el uso de VRAM sean iguales entre plataformas. Registra la GPU y el entorno que figuran en `preflight.json` al comparar corridas.

## Cómo decidir

El notebook produce `rankings/`, `metricas_retrieval/`, `generaciones/`, `seleccion_recuperadores.json` y `comparacion_final.json`. R00 es un control léxico, R01 denso BGE, R02 híbrido, R03 híbrido+BGE reranker, R04 E5, R05 Qwen3-0.6B, R06 Qwen3-4B y R07 reranker Qwen3 sobre el híbrido BGE. Los decoders Qwen2.5-7B, Qwen3-4B-Instruct-2507 y Qwen3.5-4B comparten exactamente la evidencia elegida. Los modelos tienen revisiones, parámetros y licencia registrados en `modelos_verificados.json`. La latencia informada es la de selección de pasajes más generación con rankings precalculados; no se declara como latencia integral de consulta online.

`respaldo_literal_top10` es sólo un **proxy** contra citas extraíbles del fundamento de la muestra, sin gold documental. La puntuación determinista del evaluador oficial cubre cerradas (20), citas (20) y abstención (10); el componente RAGAS (30) necesita la clave del jurado y las dependencias de `data/oficial/scripts/requirements-evaluador.txt`. No combines 50/50 con un total de 100. Los otros 20 puntos corresponden a interfaz, bitácora, vídeo y reproducibilidad. Al activar `RUN_RAGAS`, instala ese requirements y declara `OPENROUTER_API_KEY` en el entorno; el generador no utiliza esa clave.

Los puntajes obtenidos en la muestra pública son orientativos y repetidos sobre un conjunto pequeño. La decisión final debe considerar también latencia, errores de formato, abstenciones y revisión manual de citas/versión jurídica. El corpus trae restricciones documentadas en `snapshot.json`; el notebook conserva esos textos para búsqueda, pero no convierte automáticamente `vigencia: por_verificar` en vigencia certificada ni reconstruye numerales DOCX pendientes.
