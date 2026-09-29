# Preparación del corpus ampliado

Revisión del 29 de septiembre de 2026. Equipo de experimentación previsto: RTX 4090.
El usuario solicitó corregir primero los datos y posponer el notebook de modelos.
No se descargaron pesos, ejecutaron LLMs ni construyeron índices nuevos.

Consultar: [ydata de todos los textos](reports/perfil_corpus_preparado/ydata_textos_preparados.html),
[ydata de metadatos](reports/perfil_raw_despues/ydata_documentos.html) y
[auditoría de pendientes](reports/correcciones_corpus/auditoria_actualizada.md).

## Inventario y trazabilidad

La entrega original está en `data/data/raw`: **13.945 documentos**, 15.144
archivos originales y 1.060 textos derivados. El perfil inicial comprobó la
existencia, tamaño y SHA-256 de las 16.204 referencias sin discrepancias.
No encontró identificadores repetidos ni hashes de archivo duplicados.

Se agregaron **22 documentos oficiales** (17 núcleo, 5 complementarios),
incluidas SC3674-2021 y SC425-2024. El inventario actualizado tiene **13.967
documentos**. Se conservaron intactos los archivos originales de la entrega.
La revisión de identidad y los intentos fallidos están en
`reports/completar_prioritarios/verificacion.json`.

Se aplicaron **210 cambios de campos** respaldados por texto de la fuente:

- 197 números de proceso que incluían la fecha del nombre del archivo.
- Números de nueve conceptos DIAN, con sus identificadores originales en la bitácora.
- Año y título de una sentencia del Consejo de Estado: 2021, no un año 1900 inferido del identificador histórico.
- Año 2017 y numeraciones alternativas del concepto DIAN sobre procedimiento y sanciones.

El [Concepto DIAN 14116 de 2017](https://normograma.dian.gov.co/dian/compilacion/docs/concepto_tributario_dian_0014116_2017.htm)
confirma el año y el número de publicación; distingue el identificador interno
100202208–0662. No se confundió la fecha del sello con la de radicación.

El manifiesto previo se conserva en
`data/data/respaldo_manifest/manifest.8905675c3dd5a35a.json`.
Los cambios están en `reports/correcciones_corpus/cambios_20260929T115201Z.json`
y también en `correcciones_metadatos` de cada documento afectado.
Los SHA-256 suplementarios están en `data/data/SHA256SUMS.correcciones`, cuyas
rutas son relativas a `data/data`. El archivo `data/SHA256SUMS` conserva la
instantánea original; su checksum del manifiesto corresponde al respaldo.

## Correcciones de extracción

`scripts/corpus/ingesta.py`, versión `0.5.0-local`, incorpora:

- Verificación y lectura de `texto_derivado` anidado en cada archivo del manifiesto.
- OCR con procedencia; sólo asigna páginas si coinciden los bloques y las páginas del PDF.
- Extracción nativa de DOCX con tablas, referencias y cuerpos de notas. Los TXT
  anteriores sólo contenían `word/document.xml`; se detectaron 3.848 entradas de
  notas en 255 DOCX. Los sidecars se conservan y verifican, pero no sustituyen la
  extracción nativa más completa.
- Selección de una sola representación `Choice/Fallback` en Word: elimina
  duplicaciones en 17 DOCX y conserva separadores de tablas y cuadros de texto.
  Los otros 299 DOCX mantienen el mismo texto; los 7.706 marcadores de
  referencia/cuerpo de nota se conservan. Esos marcadores no equivalen a notas únicas.
- Respeto por el charset declarado del HTML y registro de bytes no decodificables.
- Retiro de navegación y pie editorial mediante selectores comprobados del Senado,
  y de un prefijo de metadatos/CSS Word reconocido. Conserva notas de vigencia y tachados.
- Reconocimiento de encabezados como `ARTICULO. 1º` y segmentación conservadora
  de conceptos/compendios, sin atribuirles automáticamente artículos propios.

El corpus exigía una ingesta externa 0.7.0 que no venía en esta entrega. Esta
implementación local corrige capacidades concretas; no se presenta como aquella versión.

## Reproducir la preparación

Desde la raíz del repositorio, con las dependencias de `requirements.txt`:

```powershell
.venv\Scripts\python.exe -m scripts.corpus.preparar_corpus --raw data/data/raw --output data/processed/corpus_preparado --workers 6
```

Guarda texto canónico y un registro por documento, vuelve a verificar los
originales al reutilizar caché y permite reanudar una interrupción. Los futuros
offsets se referirán a ese texto canónico, no a bytes del HTML/PDF. El informe de
pendientes distingue errores, omisiones expresas de la fuente y alertas que aún
necesitan revisión. No se generan chunks ni embeddings en este paso.

Para reproducir ydata-profiling sin alterar las dependencias de experimentación:

```powershell
python -m venv .venv-profiling
.venv-profiling\Scripts\python.exe -m pip install -r requirements-profiling.txt
.venv-profiling\Scripts\python.exe -m scripts.corpus.perfil_raw --raw data/data/raw --output reports/perfil_raw_despues --sample-size 320 --sample-ids reports/perfil_raw/muestra_ids.json --workers 2
.venv-profiling\Scripts\python.exe -m scripts.corpus.perfil_preparado --input data/processed/corpus_preparado --output reports/perfil_corpus_preparado
```

`reports/perfil_raw` es el perfil inicial completo de metadatos/archivos y una
muestra estratificada de 320 documentos. El perfil posterior usa exactamente
los mismos IDs para permitir comparación. La muestra no es una estimación
poblacional de calidad textual; tiene límites de tiempo y de páginas PDF
explícitos. Los 22 documentos nuevos se comprueban en la preparación completa.
El perfil `perfil_corpus_preparado` lee los textos de todos los documentos y
verifica sus hashes y longitudes. Sus palabras contadas con regex no son tokens
de un encoder/decoder. El HTML contiene estadísticas, no el texto íntegro.

## Límites y pendientes

Tener todos los archivos del manifiesto no demuestra cobertura de todo el derecho
colombiano, corrección del OCR ni vigencia jurídica. Se resolvieron 20 de las
2.865 referencias prioritarias del grafo heredado: **2.845 claves siguen ausentes**.
Esto no equivale a 2.845 fuentes cuya identidad esté confirmada; algunas citas
pueden contener números o años erróneos. El grafo se cotejó contra el inventario,
sin reconstruir sus aristas. La auditoría actual está en
`reports/correcciones_corpus/auditoria_actualizada.md` y sus versiones JSON/CSV.
No se trata cada salto de numeración o año ausente como un documento perdido.

Hay 246 documentos sin año y 400 sin número; muchos son compilaciones para las
que un valor único no corresponde. **Los 13.967 registros tienen vigencia
`por_verificar` y listas `temas` vacías**. Por tanto, aún no hay una base validada
para filtros jurídicos por materia o vigencia. Se conservan por separado las
marcas declaradas por las fuentes: no equivalen a una vigencia actual certificada.

Hay compilaciones que declaran omitir el texto de una norma; se preservan para
auditoría y se marcan no aptas para búsqueda. No se reconstruye texto legal con
un modelo. SL1972-2025 sigue sin sentencia completa localizada; un edicto no la
sustituye. T248-2025 mantiene el pendiente/exclusión documentado por la entrega.
SC1121-2018 presenta 46 páginas PDF frente a 45 bloques OCR: su paginación queda
sin verificar. Los compendios y circulares no reciben un año o número inventado.

En 90 DOCX se detectaron 783 estructuras XML de numeración automática. Se
conservó el texto de los párrafos, pero el extractor no reconstruye sus etiquetas
numéricas; las 783 estructuras no equivalen necesariamente a 783 numerales visibles.
La lista está en `reports/correcciones_corpus/docx_numeracion_automatica_pendiente.csv`.
Tampoco se certificó la fidelidad visual de imágenes, cabeceras/pies o celdas
fusionadas de Word. El detalle y la lectura de los PDF aportados están en
`reports/correcciones_corpus/contexto_documentos_usuario.md`.

La elección de encoder, decoder, chunking y guardrails se retomará en el notebook
después de revisar este estado. La investigación inicial está conservada en
`reports/investigacion_modelos.md` y `reports/investigacion_rag_juridico.md`.

Para el Bloque 0, la base recomendada es conservar el original y usar este texto
canónico con limpieza comprobada; una variante de búsqueda normalizada puede
derivarse después. No conviene sustituir la evidencia por minúsculas, fechas
reescritas o etiquetas inferidas. Los tags deberán distinguir extracción literal,
inferencia y validación, con procedencia y versión.

## Verificación de esta ejecución

La preparación final procesó **13.967/13.967 documentos**, guardó
**1.179.656.332 caracteres** y terminó con **cero errores**. La auditoría verificó
hashes, longitudes y rangos de offsets de todos los textos, sin fallos de integridad.
No encontró grupos de textos canónicos idénticos.

Se registraron 62 alertas actuales en 44 documentos distintos: 24 señales de
codificación residual, 15 casos con bytes no decodificables, 9 OCR por revisar,
8 documentos con páginas de poco texto, 1 paginación OCR no verificada,
3 omisiones declaradas y 2 fuentes restringidas por defecto previamente confirmado.
Las categorías se solapan; una alerta no demuestra por sí sola pérdida de texto.
Quedan 1.255 alertas heredadas sin reevaluación, explícitamente separadas.

**Cinco documentos quedan marcados no aptos para búsqueda**: los tres decretos
con texto omitido y las fuentes de SC5191/2020 y C264/2026 previamente restringidas.
El Decreto 1147/1999 conserva un pendiente de estructura articular. Los problemas
DOCX descritos arriba son controles complementarios, no parte de las 62 alertas.

Pasaron 41 pruebas automatizadas; se omitió una clase histórica porque faltan
sus derivados antiguos. La validación del corpus completo es adicional a esas
pruebas. Los registros finales están en `data/processed/corpus_preparado`.

El perfil posterior de metadatos/archivos verificó 16.226 referencias y confirmó
que ningún original previo fue modificado. En la muestra pareada, las señales
editoriales pasaron de 145 a 2; las dos restantes son notas de reforma y
concordancia conservadas deliberadamente, no ruido de navegación.

El **censo ydata final de los 13.967 textos** terminó sin textos ausentes,
hashes inválidos ni longitudes discrepantes. Cuenta 187.979.709 palabras por
regex, con mediana de 7.028 palabras por documento, percentil 95 de 43.183
y máximo de 617.150. Son palabras, no tokens. Estos tamaños favorecen evaluar
segmentación por estructura y recuperación de contexto superior, en lugar de
asumir que cada documento cabe en una sola entrada del encoder.

El campo conservador `estado_extraccion` marca 5.695 documentos como `extraido`
y 8.272 como `revisar`, incluyendo avisos editoriales y de formato. No representa
8.272 fallos de extracción ni debe confundirse con los 44 documentos de las
alertas específicas. Las distribuciones completas están en
`reports/perfil_corpus_preparado/metrics.json` y el HTML enlazado al inicio.
