# Interfaz

Desde la raíz del repositorio:

```powershell
python -m legalrag.cli serve
```

Abrir http://127.0.0.1:8000. Requiere un índice construido y los modelos disponibles.

La interfaz muestra respuesta, evidencia, referencias y documentos recuperados. Usa el mismo pipeline de la entrega. Para selección múltiple, escribir las cuatro opciones en líneas separadas con A), B), C) y D).

El modo `?demo=1` utiliza datos fijos y se anuncia en pantalla. La interfaz y el servidor nuevos están pendientes de ejecución en la GPU del equipo.
