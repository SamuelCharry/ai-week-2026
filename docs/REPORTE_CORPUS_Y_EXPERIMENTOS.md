# Corpus para evaluación y combinaciones prioritarias

**Proyecto:** AI Week 2026 · **Fecha:** 29 de septiembre de 2026 · **Equipo previsto:** RTX 4090.

## Decisión recomendada

Usar **`corpus_eval_v1`**, una versión documental fijada de **13.962 documentos**, para comparar configuraciones sobre la misma evidencia. El inventario original preparado conserva 13.967: cinco fuentes con defectos confirmados u omisiones declaradas quedan fuera de esta selección. No se necesita esperar a completar todo el derecho colombiano para empezar experimentos reproducibles.

La primera combinación que probaría es **BM25 + BGE-M3 + BGE-reranker-v2-m3 + Qwen2.5-7B-Instruct**, con artículo íntegro como unidad normativa, ventanas vinculadas para búsqueda, secciones para jurisprudencia, cabeceras literales, generación única y validación de citas. Es una **hipótesis priorizada**, no un ganador medido. Compararía Qwen3.5-4B como retador actual tras verificar su runtime, Qwen3-4B-Instruct-2507 como alternativa menor y los embeddings Qwen3; conservaría Salamandra como control.

“Definitivo” significa aquí **versión fija para evaluar**. No significa vigencia jurídica certificada, exhaustividad nacional, estructura de artículos validada en todos los documentos ni entrega final ya terminada. Falta construir y verificar las unidades citables y el índice. No se creó notebook, no se ejecutaron modelos ni se midió una nueva calificación.

## 1. Evidencia de todos los datos

Se usó **ydata-profiling 4.18.4**, con perfiles reales HTML/JSON y CSV reproducibles. El censo textual leyó **todos los 13.967 textos**, sin truncarlos a las primeras páginas: verificó SHA-256, longitud, palabras regex y líneas. El perfil de metadatos y archivos verificó 16.226 referencias. El perfil estructural complementario distingue encabezados candidatos de unidades jurídicas verificadas.

- [Ydata: todos los textos](reports/perfil_corpus_preparado/ydata_textos_preparados.html).
- [Ydata: estructura completa, 13.967 documentos y 89 variables](reports/perfil_estructura/ydata_estructura.html).
- [Ydata: metadatos documentales](reports/perfil_raw_despues/ydata_documentos.html).
- [Ydata: archivos originales y derivados](reports/perfil_raw_despues/ydata_archivos.html).
- [Auditoría de integridad y pendientes](reports/correcciones_corpus/auditoria_actualizada.md).
- [Distribución completa por tipo](reports/perfil_corpus_preparado/distribucion_palabras_por_tipo.csv).

La muestra pareada de 320 textos del trabajo anterior sirve para comparar limpieza antes/después; **no sustituye estos censos**. Ydata resume distribuciones, faltantes y alertas estadísticas; no certifica vigencia, interpretación jurídica, OCR ni atribución de artículos. Se desactivaron correlaciones indiscriminadas entre identificadores y categorías. No se introdujo el banco de respuestas en los datos perfilados.

| Medida del corpus preparado completo | Resultado |
|---|---:|
| Documentos / textos verificados | 13.967 / 13.967 |
| Originales / derivados referenciados | 15.166 / 1.060 |
| Fallos de integridad / errores de extracción | 0 / 0 |
| Caracteres | 1.179.656.332 |
| Palabras regex, no tokens de modelos | 187.979.709 |
| Mediana / p90 de palabras por documento | 7.028 / 30.198 |
| p95 / p99 | 43.183 / 99.412,26 |
| Máximo | 617.150, Decreto 2555/2010 |
| Documentos con más de 100.000 palabras | 138 |
| Textos canónicos exactamente idénticos | 0 grupos |

| Tipo | Documentos | Mediana de palabras | p95 |
|---|---:|---:|---:|
| Sentencia | 9.188 | 10.438 | 48.951,95 |
| Ley | 2.786 | 1.647 | 20.349,25 |
| Decreto | 1.654 | 2.714 | 22.430,05 |
| Compendio | 240 | 834,5 | 4.278,5 |
| Acto legislativo | 66 | 877 | 3.717,25 |
| Concepto | 22 | 9.149,5 | 129.935,9 |
| Circular | 6 | 31.438,5 | 122.705,25 |
| Decisión | 2 | 15.754,5 | 25.083,45 |
| Acuerdo | 1 | 13.084 | 13.084 |
| Auto | 1 | 127.572 | 127.572 |
| Constitución | 1 | 76.669 | 76.669 |

**Consecuencia principal:** las sentencias son aproximadamente el 65,8% de documentos y el 82,3% de caracteres. Un único segmentador que exija artículo a todo el corpus eliminaría o representaría mal buena parte de la evidencia. Tampoco conviene dar un documento entero al encoder: la cola de longitudes es muy grande.

![Distribución de documentos y longitudes por tipo; ambas escalas son logarítmicas](reports/reporte_evaluacion/perfil_resumen.svg)

### Estructura observada y su consecuencia

El nuevo censo estructural leyó los 13.967 textos completos; sus recuentos de
palabras/caracteres coinciden con el anterior y el snapshot no cambió.
Los 89 campos de ydata son métricas por documento, no una anotación humana de artículos.

| Hallazgo medido | Interpretación | Decisión experimental |
|---|---|---|
| 13.227 documentos, el 94,70%, quedan en un solo bloque al dividir por líneas vacías | La normalización no conserva párrafos originales universales | No usar `split("\\n\\n")` como segmentador general |
| 9.132/9.188 sentencias tienen alguna sección judicial candidata; 7.110 muestran candidatos de antecedentes, consideraciones y decisión | Hay señal aprovechable, pero no certificación de la estructura o del fallo | Segmentador jurisprudencial con fallback y revisión de límites |
| 7.658 sentencias contienen 61.429 encabezados de artículo candidatos | Muchos pueden pertenecer a normas citadas, notas o transcripciones | No convertirlos en “artículos de la sentencia” |
| 4.503/4.516 documentos normativos y otros actos tienen encabezados articulares candidatos | Hay señal para construir padres normativos; requiere atribución | Usar anclas y jerarquía del original y verificar excepciones |
| 231.514 candidatos articulares en todo el corpus, 169.443 en el grupo normativo/otros actos | Incluyen repeticiones, versiones y citas; no son artículos únicos verificados | No usar este total como tamaño definitivo del índice ni cobertura jurídica |
| 1.043/4.516 normas y otros actos tienen rótulos candidatos repetidos; 303 tienen algún intervalo entre candidatos de más de 2.048 palabras | Igual número puede corresponder a cita, versión o disposición distinta | No deduplicar sólo por rótulo; verificar padres y presupuesto con tokenizador |
| 3.571 documentos con candidatos de libro/título/capítulo | Jerarquía literal disponible en parte del corpus | Comparar cabecera literal frente a sólo contenido |
| 1.369 documentos con señales de versión anterior; 934 en el grupo normativo/otros actos | Pueden coexistir texto histórico, vigente, notas y menciones | Mantener versiones separadas y no confundir señal textual con vigencia |
| 1.311 documentos con tablas HTML registradas; 2.399 con líneas candidatas de tabla | Medidas distintas, solapadas; no sumarlas | Preservar encabezados de tabla y relación entre celdas al segmentar |
| 240 compendios; 34 con candidatos articulares y 61 con candidatos de sección judicial | Estructura heterogénea; ni norma única ni sentencia única | Recuperar como compilación y seguir procedencia a la fuente cuando exista |

Las reglas están ancladas a líneas y pueden omitir variantes de OCR o encabezados
pegados. Un candidato no prueba titularidad, vigencia ni suficiencia jurídica;
su ausencia tampoco prueba pérdida de contenido. Los intervalos entre candidatos
incluyen notas, citas y el tramo final hasta EOF. Los cuantiles por documento no
equivalen a cuantiles globales de artículos. El detector final 1.1.0 exige señales
de encabezado y excluye algunas referencias cortadas al comienzo de línea;
los conteos preliminares 1.0.0 quedan reemplazados. Las
[13 normas/actos sin candidato](reports/perfil_estructura/normativa_sin_candidato_articular.csv)
requieren revisión por tipo; no son 13 fuentes sin texto demostradas.
[Método y límites](reports/perfil_estructura/LEEME.md).

Hay además cuatro sentencias enteras en una sola línea, hasta 45.216 palabras
(`sentencia_cc_c489_2019`). Por tanto, tampoco basta segmentar por cada salto de
línea: se necesita un fallback con límites comprobados y conservación de offsets.

### Metadatos utilizables y límites

| Campo | Estado observado | Uso recomendado |
|---|---|---|
| `doc_id`, URL, título, fuente, fecha de consulta, hash | Identidad/procedencia conservadas | Identidad de documentos, auditoría y deduplicación |
| `tipo`, órgano, número, año | Estructurados; 246 documentos sin año y 400 sin número | Restricción explícita del usuario con respaldo; admitir nulos y alias históricos |
| `areas` | Presentes en todos los documentos; 3.981 multiarea | Señal blanda para ordenar/estratificar; validar antes de excluir por área |
| `temas` | Listas vacías en 13.967 documentos | No hay taxonomía temática fina utilizable todavía |
| `vigencia` | `por_verificar` en todos | No filtrar como “vigente” ni atribuir actualidad por defecto |
| `vigencia_fuente` | 12.877 sin marca, 1.026 con notas, 39 inexequible, 25 derogada | Señal declarada por fuente; revisar alcance por artículo y fecha |
| `nivel` | 11.232 núcleo y 2.735 complementarios | Ablación de alcance; no ocultar complementarios de forma irrevocable |
| `estado_extraccion` | 5.695 extraído, 8.272 revisar | Aviso conservador, no conteo de fallos |

`areas` y `temas` son campos diferentes: la ausencia de temas finos **no implica ausencia de áreas**. Una etiqueta heredada del grafo o del epígrafe puede ser útil sin constituir clasificación jurídica comprobada. Las marcas de vigencia del publicador tampoco acreditan consolidación actual.

| Área declarada | Documentos con etiqueta | Peso de preguntas en el enunciado |
|---|---:|---:|
| Constitucional | 8.722 | 12,86% |
| Administrativo | 3.796 | 11,90% |
| Penal | 1.996 | 11,80% |
| Procesal | 1.755 | 10,65% |
| Comercial | 431 | 9,98% (comercial y sociedades) |
| Civil | 516 | 9,79% |
| Familia | 504 | 8,93% |
| Tributario | 825 | 8,83% |
| Laboral | 909 | 8,35% |
| Mercados | 364 | 6,91% |

Las etiquetas se solapan y no se deben sumar como particiones. Estos números
priorizan auditoría de recuperación por área, no demuestran infracobertura ni
permiten convertir “documentos por etiqueta” en porcentaje de preguntas resolubles.
Hay 3.390 registros con `areas_por_epigrafe=true`; en 10.577 el campo está ausente,
lo que no significa `false`. El nuevo censo lee ese booleano correctamente desde
el manifiesto; el aplanado anterior como lista no sirve para analizar ese flag.
Se detectaron 102 etiquetas de fuente Senado con URL de Función Pública: se
registran como diferencias de alias/procedencia, sin declararlas identidades falsas.

La limpieza conserva mayúsculas, tildes, cifras, negaciones, excepciones y notas de reforma. Se recuperaron notas presentes en 255 DOCX y se corrigieron representaciones duplicadas en 17. No se reconstruyeron números de lista que no aparecen como texto visible. Las líneas del canónico no equivalen universalmente a párrafos de la fuente original.

## 2. Versión fija para comenzar a evaluar

El [manifiesto de evaluación](data/releases/corpus_eval_v1/corpus_manifest.json) contiene documento, procedencia, áreas heredadas, ruta del texto, hash, licencia declarada y restricciones. [snapshot.json](data/releases/corpus_eval_v1/snapshot.json) fija su identidad:

`7eabf300335b23be5541540c34b1828c15c0a05b3f65c595a22ff8605b8ebd77`

| Conjunto | Cantidad | Tratamiento |
|---|---:|---|
| Preparado completo | 13.967 | Se conserva íntegro para auditoría |
| Selección evaluable documental | 13.962 | Base común de los experimentos |
| Excluidos confirmados | 5 | Decretos 1497/1993, 1940/1992, 543/1993; SC5191/2020 y C264/2026 |
| Evaluables con restricciones específicas | 130 | Revisar evidencia afectada; no inventar numerales, páginas ni articulado |
| Evaluables sin esas restricciones específicas | 13.832 | No implica certificación jurídica ni ausencia de avisos generales |

Las restricciones incluyen 90 DOCX con numeración automática no reconstruida, OCR, codificación y Decreto 1147/1999 sin unidad articular verificada. Las categorías se solapan. La auditoría anterior contaba 44 documentos con alertas específicas del preparador; 130 incorpora además los controles estructurales y excluye los cinco ya separados. [Restricciones por documento](data/releases/corpus_eval_v1/restricciones.json).

La versión fija **referencia los textos preparados**; no crea otra copia de 1,2 GB ni los vuelve físicamente inmutables. Antes de cada lote:

```powershell
.venv\Scripts\python.exe -m legalrag.ingestion.fijar_corpus_evaluacion --version corpus_eval_v1 --verificar
```

El comando valida el inventario y cada texto seleccionado. Si cambia un hash, se debe crear otra versión, reconstruir los índices afectados y separar sus resultados. El script no sobrescribe una versión existente. Los originales, el manifiesto previo y las bitácoras de 210 correcciones de campos y 22 incorporaciones permanecen disponibles.

**Cobertura pendiente:** 2.845 claves prioritarias del grafo heredado aún no están incorporadas; no son 2.845 identidades jurídicas necesariamente correctas. SL1972/2025 y T248/2025 siguen pendientes de texto íntegro. No conviene convertir toda cita ausente en una descarga automática ni reconstruir normas con un LLM.

## 3. Qué maximiza la nota según el enunciado

Fuente: [enunciado.pdf](enunciado.pdf), páginas impresas 4–10 y anexos, especialmente pp. 8–9, 12 y 14. La página física del PDF corresponde a la impresa más uno.

| Componente | Puntos | Prioridad de ingeniería |
|---|---:|---|
| Exactitud en preguntas cerradas | 20 | Recuperar la disposición decisiva; resolver opción y descarte con evidencia |
| RAGAS `answer correctness` | 30 | Responder la pregunta, conservar excepciones y evitar afirmaciones no sustentadas |
| Citas | 20 | Recall de fundamento y respaldo en los primeros diez pasajes |
| Abstención | 10 | Calibrar cobertura frente al error; abstenerse siempre sólo da 5 puntos |
| Interfaz | 10 | Consulta funcional 4, evidencia/citas visibles 3, identidad Software Colombia 3 |
| Corpus | 5 | Inventario 2, selección justificada 1,5, reconstrucción desde URLs 1,5 |
| Video | 3 | Demostración y decisiones, máximo cinco minutos |
| Reproducibilidad | 2 | Comando único desde entorno limpio sobre muestra |

La expresión descrita es `20A + 30R + 20C + 10U + interfaz + corpus + video + reproducibilidad`. En citación, una referencia que está en el fundamento y en evidencia obtiene valor 1; en fundamento pero sin evidencia, 0,5; fuera del fundamento pero respaldada, 0 sin penalización; fuera de ambos recibe penalización doble. **No inventar denominadores, clipping ni agregación:** no quedan completamente definidos en el PDF.

No se deben retirar respuestas correctas sólo para aumentar abstención. El beneficio de un umbral depende de todos los componentes, no sólo de la utilidad 1/0,5/0. Seleccionar umbrales en desarrollo mediante puntaje oficial cuando esté disponible y mostrar además curva riesgo–cobertura, errores y citas sin respaldo.

### Reglas obligatorias y decisiones derivadas

- Decoder abierto con **máximo estricto de 8.000.000.000 parámetros**, incluso cuantizado; encoder abierto; temperatura 0. La cuantización reduce memoria, no parámetros.
- Ningún modelo cerrado como generador, reformulador, reranker o productor de datos sintéticos de apoyo. El juez oficial es una herramienta de evaluación separada.
- No indexar el banco de preguntas ni material que contenga respuestas esperadas; tampoco editar manualmente respuestas generadas. Mantener corpus y evaluación separados.
- Norma segmentada a nivel de **artículo íntegro**. Las ventanas son unidades auxiliares de búsqueda, vinculadas al artículo recuperable completo.
- Seleccionar como máximo **diez pasajes finales** que realmente respalden citas y afirmaciones. La búsqueda interna puede recuperar más candidatos.
- JSON según formato, sin inventar un esquema común diferente: `multiple_choice`, `semi_open`, `open_ended`, más identidad, abstención y pasajes. Campo `respuesta` de semiabiertas: 3–5 oraciones y máximo 150 palabras; campo `analisis` de abiertas: 5–8 oraciones.
- Índice congelado al entregar y regeneración consistente de 2–3 respuestas. Registrar revisiones, semillas, desempates y configuración, además de temperatura 0.
- Para 992 preguntas en seis horas, el presupuesto total medio es **21,77 segundos por pregunta**. Debe incluir recuperación, reranking, generación, reintentos y escritura. Es un presupuesto, no una latencia medida.

Los pesos por área son: constitucional 12,86%; administrativo 11,90%; penal 11,80%; procesal 10,65%; comercial y sociedades 9,98%; civil 9,79%; familia 8,93%; tributario 8,83%; laboral 8,35%; mercados 6,91%. Evaluar por estas áreas y por formato, con promedio ponderado y macro. Tener más archivos constitucionales no demuestra mejor cobertura de preguntas constitucionales.

## 4. Bloques que conviene comparar

| Bloque | Primera opción | Alternativa de valor | Qué evitar inicialmente |
|---|---|---|---|
| 0. Preparación | Canónico limpio, identidad literal y procedencia | Vista léxica con alias verificables; áreas como señal blanda | Minúsculas sobre evidencia, quitar negaciones, inventar vigencia o tags |
| 1. Segmentación | Por tipo: artículo para normas; sección/bloque para sentencias | Ventanas hijas + cabecera jerárquica literal + retorno al padre | Un artículo ficticio por sentencia; cortar una norma y citar el corte como artículo íntegro |
| 2. Recuperación | BM25 + denso, fusionados con RRF | Enrutamiento blando por norma exacta/tipo y un reintento acotado | Sumar scores crudos incompatibles o descartar por tags inferidos |
| 3. Reordenamiento | BGE-reranker-v2-m3 sobre candidatos | Qwen3-Reranker-0.6B; control sin reranker | Umbral universal 0,85 o promesa fija de 2–3 segundos |
| 4. Generación | Una llamada con evidencia y formato explícitos | Una verificación selectiva, disparada por evidencia insuficiente/conflicto | Multiagente incondicional; presentar prompting como Self-RAG entrenado |
| 5. Validación | JSON, IDs, artículos, pertenencia a evidencia y soporte de afirmaciones | Calibración de abstención y validación temporal específica | Borrar una cita y conservar la afirmación que dependía de ella |

### Segmentación propuesta, sin ejecutarla todavía

**Normas:** construir padre con `doc_id`, identificador literal de artículo, encabezado, offsets y texto completo; distinguir disposiciones vigentes, anteriores y comentarios. Una mención a otro artículo dentro de una nota no crea un artículo propio. Cuando el padre sea largo, indexar ventanas hijas; agrupar resultados por padre, sin llenar los diez pasajes con duplicados del mismo artículo. Si el artículo completo no cabe, registrar desbordamiento y probar más contexto o evidencia alternativa; no truncarlo silenciosamente.

**Sentencias y autos:** conservar radicado/identidad y separar antecedentes, problema, consideraciones y decisión cuando se detecten con evidencia. El padre será una sección o bloque coherente, no necesariamente toda una sentencia de decenas de miles de palabras. Los títulos candidatos ayudan a buscar, pero no certifican que exista ratio decidendi ni sentido del fallo correctamente extraído.

**Compendios, conceptos y circulares:** usar secciones y numerales comprobables; distinguir una compilación de una providencia individual. No exigir el campo artículo para permitir su recuperación. Las tablas deben conservar encabezados con las filas; las notas deben quedar vinculadas a su unidad, no perderse al cortar.

Inicio propuesto: ventanas de búsqueda de **384 tokens**, solapamiento **64**, cabecera literal corta incluida en el presupuesto. Comparar después 768/128 con BGE/Qwen. Para comparar encoders causalmente, fijar los mismos spans y comprobar que caben con **todos** sus tokenizadores, incluido E5 de 512 tokens; medir y reportar truncación cero. Palabras regex y caracteres no sustituyen esa tokenización.

## 5. Combinaciones prioritarias

Todas comparten la versión del corpus, política de citas, máximo diez evidencias y temperatura 0. Las fichas oficiales acreditan capacidades y licencias, no rendimiento en este corpus.

| Opción | Recuperación | Reranker | Decoder | Motivo y coste a contrastar |
|---|---|---|---|---|
| **A — primera prueba** | BM25 + BGE-M3, RRF | BGE-reranker-v2-m3 | Qwen2.5-7B-Instruct | Base multilingüe con búsqueda literal; evaluar calidad de respuesta y JSON |
| **B — presupuesto de generación** | BM25 + Qwen3-Embedding-0.6B, RRF | BGE-reranker-v2-m3 | Qwen3-4B-Instruct-2507 | Menos parámetros de generación; comprobar si conserva calidad y mejora margen de tiempo |
| **C — recuperación más costosa** | BM25 + Qwen3-Embedding-4B, RRF | Qwen3-Reranker-0.6B | Qwen3-4B-Instruct-2507 | Candidato condicionado a mejora demostrada de recuperación; más coste de indexación y vectores |
| **D — retador actual de generación** | BM25 + BGE-M3, RRF | BGE-reranker-v2-m3 | Qwen3.5-4B, sólo texto, thinking desactivado | Comparar contra A con evidencia idéntica; primero validar un entorno compatible |
| Control del proyecto | BM25 + BGE-M3 | BGE-reranker-v2-m3 | Salamandra-7B-Instruct | Mantener un decoder en español conocido por el proyecto; resultados antiguos no cuentan como prueba nueva |
| Control de encoder | BM25 + multilingual-e5-large | BGE-reranker-v2-m3 | El decoder fijado | Contraste con prefijos `query:`/`passage:` y límite de entrada más corto |

BGE-M3 admite representación densa, dispersa y multivector; para la primera prueba usar sólo su salida densa junto a BM25, sin cambiar simultáneamente todos sus modos. No necesita los prefijos de E5. [Ficha BGE-M3](https://huggingface.co/BAAI/bge-m3).

Qwen2.5-7B-Instruct declara Apache-2.0 y soporte multilingüe/estructurado; su contexto efectivo debe configurarse según runtime, sin asumir que el máximo anunciado cabe en la 4090. [Ficha Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). BGE-reranker-v2-m3 declara Apache-2.0; sus scores ordenan relevancia, no son probabilidades jurídicas calibradas. [Ficha del reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3).

| Decoder inicial | Parámetros aprendidos verificados | Licencia declarada de pesos |
|---|---:|---|
| Qwen2.5-7B-Instruct | 7.615.616.512 | Apache-2.0 |
| Qwen3-4B-Instruct-2507 | 4.022.468.096 | Apache-2.0 |
| Salamandra-7B-Instruct | 7.768.117.248 | Apache-2.0 |
| Qwen3.5-4B, modelo completo incluido componente visual/MTP | 4.659.865.088 | Apache-2.0 |

Qwen3-8B (8.190.735.360) y Llama-3.1-8B (8.030.261.248) exceden el límite estricto;
no los incluiría sin aclaración oficial. La [investigación verificada](reports/reporte_evaluacion/investigacion_actualizada.md)
y el [catálogo con revisiones y licencias](reports/reporte_evaluacion/modelos_verificados.json)
incluyen evidencia de conteo, plantillas, contextos y diferencias entre licencia
de pesos y código. E5 exige sus prefijos; Qwen Embedding usa instrucción de consulta
y pooling propio, aunque funcione como encoder de búsqueda. Qwen3 requiere un
entorno compatible separado del `transformers==4.48.3` actual.

**Actualización de candidatos 2026:** Qwen3.5-4B admite entrada sólo textual y
thinking desactivado. Su arquitectura híbrida necesita soporte específico:
la ficha recomienda Transformers main y se verificó implementación en versiones
5.2/5.3; actualizar sólo a 4.51 no basta. Su carga completa ronda 8,68 GiB en
los dtypes publicados, antes de memoria de ejecución. No aplicar la fórmula KV
de un transformer convencional a todas sus capas DeltaNet. Probar temperatura 0
por exigencia del enunciado aunque la ficha sugiera sampling. No se midió calidad
jurídica ni velocidad. [Ficha y artefactos oficiales](https://huggingface.co/Qwen/Qwen3.5-4B).

Las combinaciones completas sirven como destinos de comparación; **no atribuir la diferencia A→B a un solo modelo**, porque cambian encoder y decoder. El protocolo siguiente cambia una pieza a la vez.

## 6. Orden de experimentos para gastar cómputo donde aporta

**Fase 0: contrato.** Verificar snapshot, identidad de unidades citables, ausencia del banco en índices, campos por tipo y pasajes completos. El paquete oficial ya está en `data/oficial`; ejecutar su evaluador sobre cada entrega antes de declarar una nota. No descargar todas las familias de pesos a la vez.

**Fase 1: recuperación, sin generador.** Con las mismas consultas y unidades:

| ID | Cambio respecto al anterior/control | Qué medir |
|---|---|---|
| R00 | BM25 solo, control diagnóstico | Recall de documento y unidad, errores de literalidad |
| R01 | BGE-M3 denso solo | Ganancia semántica frente a coincidencia literal |
| R02 | R00 + R01, RRF con constante inicial 60 | Complementariedad; recall antes de reranking |
| R03 | R02 + BGE-reranker-v2-m3 | Recall/nDCG en las diez evidencias finales |
| R04 | R03, cambiar sólo encoder por E5-large | Alternativa de representación de entrada corta |
| R05 | R03, cambiar sólo encoder por Qwen3-Embedding-0.6B | Alternativa pequeña de embeddings |
| R06 | Mejor recuperación, cambiar sólo encoder por Qwen3-Embedding-4B | Sólo si el incremento compensa coste |
| R07 | Mejor recuperación, cambiar sólo reranker por Qwen3-0.6B | Calidad/latencia de reordenamiento |

Inicio de búsqueda: top 100 BM25 y top 100 denso, unión y RRF, reranking de top 50 candidatos; deduplicar por padre y seleccionar hasta diez evidencias. Son **valores iniciales para probar**, no óptimos medidos. Informar recall de documentos, de artículos/unidades y de respaldo de cita; no confundirlos. No usar como verdad toda cita regex hallada en un documento.

**Fase 2: decoder.** Fijar los pasajes del mejor retriever y comparar Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507 y Salamandra-7B-Instruct con el mismo contenido y límites de salida. Añadir Qwen3.5-4B como retador prioritario tras una prueba mínima de carga, plantilla, generación determinista y JSON en su entorno compatible. Usar plantillas oficiales de cada modelo. Empezar con contexto 8.192; registrar overflow, longitud útil y formato válido. Evaluar más contexto sólo para los candidatos que lo necesitan y soportan.

**Fase 3: ablaciones dirigidas.** Cabecera literal activada/desactivada; hijo pequeño/mediano; padre frente a ventana como representación de búsqueda manteniendo evidencia normativa íntegra; área como señal blanda frente a sin área; núcleo frente a núcleo+complementarios; 13.962 frente al subconjunto de 13.832 sin restricciones específicas; reranker frente a sin reranker; una llamada frente a verificación selectiva. Cada contraste mantiene lo demás fijo. La variante estricta puede perder cobertura: no asumir que por ser menor mejora.

**Fase 4: abstención y cierre.** Ajustar umbrales sobre desarrollo; distinguir falta de recuperación, conflicto temporal, evidencia contradictoria y fallo de formato. Una sigmoid del reranker no es confianza de respuesta. Congelar política y ejecutar evaluación retenida una sola vez tras la selección. Conservar fallos, denominadores, respuestas originales y tiempos, no sólo el mejor puntaje.

No ejecutar el producto cartesiano de todos los bloques. Promover como máximo dos recuperadores a la fase de generación. Con muestra pequeña, mostrar conteos y discrepancias por pregunta y, cuando sea viable, intervalos pareados; una mejora de un ítem no acredita un vencedor robusto. No seleccionar parámetros usando el conjunto de prueba final ni claves como insumo del modelo.

El [plan legible por máquina](../configs/plan_evaluacion_corpus_v1.json) registra
esta secuencia, restricciones y métricas. Es un plan **no ejecutado**, no una
configuración compatible automáticamente con el runtime antiguo.

## 7. RTX 4090: plan de memoria y tiempo

No se midió inferencia en esta GPU durante este trabajo. Separar **parámetros**, memoria teórica de pesos y memoria total real. Pesos FP16 ocupan aproximadamente `2 × parámetros` bytes; 4 bits ideales, `0,5 × parámetros`, antes de escalas, formatos mixtos, activaciones, caché KV y runtime.

Construir embeddings por lotes, guardar índice en RAM/disco y liberar el encoder antes de la generación si hace falta. El reranker y decoder comparten memoria: cargar por fases puede ser mejor que mantener todo residente. Empezar el decoder cuantizado en 4 bits, lote de generación 1, y medir cuantización como factor experimental, con revisiones y hashes del artefacto.

Para vectores float32, un millón de chunks de 1.024 dimensiones requiere **3,81 GiB sólo de vectores**; de 2.560 dimensiones, **9,54 GiB**. No son estimaciones del número real de chunks ni memoria completa de FAISS. BGE-M3/E5-large generan 1.024 dimensiones; Qwen3-Embedding-4B permite comparar dimensiones menores documentadas antes de aceptar todo su coste. Medir también RAM, tamaño del índice y tiempos de construcción.

Registrar tiempo extremo a extremo p50/p95/media, tokens de entrada/salida, consumo máximo de VRAM, overflow, reintentos y errores. No extrapolar seis horas desde una única pregunta corta. Dar prioridad a reducir evidencia redundante y reintentos antes de incorporar dos agentes en todas las consultas.

## 8. Investigación transferible y licencias

| Trabajo / organización | Qué aporta aquí | Límite de transferencia |
|---|---|---|
| [LegalBench-RAG — ZeroEntropy](https://github.com/zeroentropy-ai/legalbenchrag) | Evaluación del fragmento que contiene evidencia | Código MIT; licencias de datasets separadas; dominio/idioma distintos |
| [ALCE — Princeton NLP](https://github.com/princeton-nlp/ALCE) | Separar respuesta correcta de citas completas y respaldadas | Código MIT; no demuestra precisión colombiana |
| [BGE — BAAI/FlagOpen](https://github.com/FlagOpen/FlagEmbedding) | Recuperación multilingüe y reordenamiento abierto | Evaluar corpus propio, sin importar resultados de otros benchmarks |
| [Haystack — deepset](https://docs.haystack.deepset.ai/docs/automergingretriever) | Relación hijos/padres y recuperación jerárquica | Auto-merging exige una condición de hijos recuperados; no equivale a cualquier parent-child |
| [RAG regulatorio — Universidad de los Andes](https://aclanthology.org/2025.regnlp-1.5/) | Comparaciones de recuperación y adaptación de dominio | No se verificó un sistema abierto equivalente para este corpus |
| [Terminología jurídica — OEG-UPM](https://aclanthology.org/2025.ldk-1.16/) | Expansión terminológica conservadora | Español de otra jurisdicción; repositorio público no implica licencia abierta |
| [CRAG](https://arxiv.org/abs/2401.15884) | Reintento/refinamiento cuando la recuperación es insuficiente | Código/artefactos requieren revisión de licencia; web dinámica compromete reproducibilidad |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | Reflexión y crítica integradas en entrenamiento e inferencia | Escribir etiquetas en un prompt no reproduce el método entrenado |

Para este proyecto adaptaría de CRAG **un único reintento sobre el corpus congelado**, con modelo abierto y disparador medible. Evitaría búsqueda web dinámica en la entrega inicial: el PDF no la prohíbe universalmente, pero las evidencias pueden cambiar al regenerar. La autoverificación propuesta se llamará así; no se presentará como una implementación de Self-RAG.

Las licencias de pesos, código, datasets y ediciones del corpus deben registrarse por separado. Un texto accesible o un repositorio público no basta. El corpus conserva condiciones declaradas por fuente y no relicencia las notas de terceros; el snapshot local no equivale a autorización de publicación de todos sus componentes.

## 9. Bloqueos concretos antes de la evaluación oficial

1. **Actualización del 29 de septiembre:** el paquete oficial ya está en `data/oficial`, incluidos `evaluate.py`, `citations.py`, `common.py` y el esquema. El notebook [E06](../notebooks/experimentos/e06_corpus_definitivo.ipynb) usa ese evaluador. Aún no hay corridas nuevas de modelos ni nota oficial para `corpus_eval_v1`.
2. **Índices anteriores incompatibles:** `configs/indice.json` apunta a `data/processed/corpus`, correspondiente al trabajo anterior de 386 documentos. Rechazar su reutilización con este snapshot salvo reconstrucción y verificación completa de manifiesto/modelo/tokenizador.
3. **Filtro por tipo a adaptar:** `src/legalrag/indexing/indice.py` rechaza unidades sin artículo si no son sentencia/auto. Excluiría conceptos y compendios válidos; adaptar contrato por tipo antes de construir el índice nuevo.
4. **Abstención cerrada:** el proyecto documenta un conflicto entre `null` y el esquema oficial histórico. El esquema actual no está presente; no considerar resuelto un validador que exceptúa el error. Contrastar con el paquete real y registrar la decisión.
5. **Citación y longitud:** las validaciones locales por coincidencia de norma no prueban artículo correcto ni apoyo sustantivo. Añadir control de evidencia/unidad y límites de oraciones/palabras; hoy algunas restricciones sólo están en el prompt.
6. **Entrega publicable:** quedan índice, revisión de redistribución por fuente/edición, LICENSE, paquete descargable durante 30 días, interfaz funcional y comando único en entorno limpio. Tener el corpus local no asegura los 20 puntos de entregables.

El enunciado tiene inconsistencias: menciona 289 cerradas, pero la tabla y la resta de la muestra darían 290; anuncia cuatro secciones de CORPUS.md y enumera tres; sugiere modelos “8B” que exceden el límite numérico estricto. Mantener el límite estricto, usar IDs/composición reales del paquete oficial y documentar las discrepancias. El informe final de entrega tiene máximo tres páginas; **este reporte es un documento interno de decisión más extenso**.

## 10. Criterio de promoción

Una variante se promueve si cumple todas las reglas, conserva trazabilidad y mejora la métrica oficial o un diagnóstico explícitamente provisional, sin ocultar citas sin respaldo, errores de esquema o desbordamientos. Elegir entre variantes con mejoras pequeñas considerando tiempo total y estabilidad. No hay evidencia local para prometer una calificación ni un modelo ganador antes de ejecutar estos experimentos.

La siguiente actividad es implementar las unidades por tipo y reconstruir la recuperación para `corpus_eval_v1`; luego ejecutar R00–R05 y seleccionar. El notebook permanece pendiente por instrucción del usuario.
