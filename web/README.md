# Interfaz

Interfaz del equipo P34K para consultar derecho colombiano y mostrar respuestas con sus pasajes. El backend aún no está conectado.

## Ejecutar

Desde la raíz del repositorio, con Python 3 instalado:

```powershell
python -m http.server 8766 --directory web --bind 127.0.0.1
```

Abrir [la interfaz](http://127.0.0.1:8766/). Se detiene con Ctrl + C en la terminal. Si el puerto está ocupado, usar otro en el comando y en la dirección.

No requiere Node ni compilación. Hay que servir los archivos por HTTP porque el JavaScript usa módulos.

## Demo

Abrir [el modo demo](http://127.0.0.1:8766/?demo=1), escribir una pregunta y pulsar Consultar.

- El selector permite recorrer los tres formatos.
- Una pregunta que contenga `abstención` muestra ese estado.
- Las respuestas usan documentos ficticios sobre una ficha azul. Las citas y puntuaciones son datos fijos para revisar la interfaz.
- Para salir de la demo, quitar `?demo=1` de la dirección y recargar.

En modo normal, las consultas se envían al servicio. Si no está disponible, se muestra un error y la pregunta se conserva.

## Conexión con el backend

La interfaz espera `POST http://127.0.0.1:8000/preguntar` con este cuerpo:

```json
{"pregunta": "Texto de la consulta", "formato": "semi_open"}
```

Los formatos admitidos son `semi_open`, `open_ended` y `multiple_choice`.

Este contrato es provisional. La respuesta esperada es un objeto con `respuesta` y, opcionalmente, `traza`. También se admite el objeto de respuesta sin envoltorio.

| Objeto | Campos |
|---|---|
| Respuesta común | `formato`, `abstencion`, `pasajes_recuperados` |
| Semiabierta | `respuesta`, `referencia_legal`, `palabras_clave` |
| Abierta | `marco_normativo`, `analisis`, `jurisprudencia`, `conclusion` |
| Selección múltiple | `respuesta_correcta`, `justificacion`, `descarte_opciones` |
| Pasaje recuperado | `doc_id`, `texto`, `score`, `inicio`, `fin` y `fragmento_id` recomendado |

`traza` puede incluir `citas_respaldadas`, `citas_sin_respaldo`, `pasajes`, `corpus`, `modelo`, `k`, `segundos`, `temperatura` y `semilla`. Los dos últimos se muestran solo si el servicio los devuelve como números.

Las citas usan `norma` y `articulo`. Los metadatos de cada pasaje incluyen `norma`, `articulo`, `url` y `score`. Se relacionan con el pasaje mediante `fragmento_id`, `doc_id` o `rank`, sin ambigüedad. El corpus usa `norma_key`, `norma`, `doc_id` y `url`.

Los estados de las citas dependen de los datos recibidos. La interfaz no comprueba por sí sola la vigencia ni la corrección jurídica.

Para cambiar la dirección del servicio, definir esto en `index.html` antes de cargar `js/app.js`:

```html
<script>window.P34K_API = "http://127.0.0.1:8000"</script>
```

Si la interfaz y el backend usan puertos distintos, el servicio debe permitir el origen de la interfaz mediante CORS. La integración se debe comprobar cuando exista el endpoint.
