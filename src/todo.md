# Resumen rápido: Faltantes para Producción

Este documento detalla los bloques principales pendientes para un despliegue robusto del proyecto. Se priorizan acciones concretas y se mapean al repositorio.

---

## Checklist Priorizada

### 1. **Harden API** _(Alta prioridad)_
**Pendiente en `api.py`:**
- Endpoint `/health` y `/metrics`.
- Mejor validación de inputs (ej. límite de tamaño para basket) y manejo de errores.
- Endpoint admin para recargar modelos sin reiniciar.
- Logs estructurados por request y nivel de log.

**Acciones concretas:**
- Añadir `@app.get("/health")` que devuelva status y timestamps.
- Usar validadores Pydantic en `PredictRequest` (máx. items).
- Implementar `reload_models()` para volcar caches y recargar modelos en memoria.

---

### 2. **Contenerización y Runtime de Producción** _(Alta)_
**Estado actual:** Dockerfile existe.

**Verificar:**
- Imagen base ligera (`python:3.11-slim`).
- Instalación de dependencias y copia de modelos.
- Arranque con `gunicorn` + `uvicorn.workers.UvicornWorker` o `uvicorn --workers`.

**Acciones concretas:**
- Dockerfile: usar gunicorn con 2-4 workers, logging a stdout.
- Probar build/run local.
- Montar `models` como volumen si se actualizan fuera de la imagen.

---

### 3. **CI/CD** _(Alta)_
**Pendiente:** Workflows para tests, linters, build y push de imagen.

**Acciones concretas:**
- GitHub Actions: ejecutar `python -m pytest`, `docker build`, `docker push` a registry.
- Opcional: despliegue a Cloud Run / AWS ECS / Azure Container Instances.

---

### 4. **Monitorización y Logging** _(Media)_
**Pendiente:** Exportación de métricas (latencia, request count, errores).

**Acciones concretas:**
- Instalar `prometheus_client` y exponer `/metrics`.
- Añadir contadores e histogramas en wrappers de endpoints.

---

### 5. **Optimización de Inferencia NCF** _(Alta técnica)_
**Problema:** `predict_ncf` puntúa todo el vocabulario en CPU por petición.

**Opciones:**
- Candidate Sampling: preselección heurística (populares, cesta, categorías).
- Precomputar embeddings y usar ANN (FAISS) para queries rápidas.
- Servir inferencia desde GPU o usar ONNX + TensorRT.
- Cache de recomendaciones frecuentes.

**Acción inicial:** Implementar ruta rápida con top-N populares + filtro por basket; scoring completo para batch offline.

---

### 6. **Pipelines de Datos y Retraining** _(Media)_
**Pendiente:** ETL para train/val y jobs programados para retrain.

**Acciones concretas:**
- Crear DAG con Prefect/Airflow para features (migrar notebooks a scripts), entrenar y subir artefactos a model registry (S3/GCS).

---

### 7. **Tests y QA** _(Alta)_
**Pendiente:** Tests unitarios e integración.

**Acciones concretas:**
- Añadir tests en `tests` (ej. `test_api.py`) para endpoints `/predict/*`, mocks de modelos y archivo `products.csv` pequeño.
- Performance smoke test: latencia y throughput del endpoint.

---

### 8. **Seguridad y Operaciones** _(Media)_
**Pendiente:** Autenticación (API key), rate limiting, manejo seguro de secretos.

**Acciones concretas:**
- Integrar middleware FastAPI para rate limit y autenticación por header Authorization.
- Usar secretos en CI (GitHub Secrets) y montar secretos en runtime.

---

### 9. **Versionado y Model Registry** _(Media)_
**Pendiente:** Versionado formal de modelos y metadatos.

**Acciones concretas:**
- Guardar checkpoints con metadata (training args, metrics) y usar S3/GCS/artifact registry.
- Añadir `models/model_manifest.json`.

---

## Prioridad Inmediata — Qué Implementar Hoy

- Añadir `/health` y endpoint para recargar modelos en `api.py`.
- Probar Dockerfile actual: build + run local.
- Añadir test minimal para `utils.load_product_names`.

**Comandos de ejemplo (PowerShell):**
```powershell
# Ejemplo de comandos para probar endpoints y build
```

---

## Riesgos y Edge Cases

- **Model large:** checkpoint NCF y vocab size — RAM/CPU/latency.
- **Concurrency:** inferencia sobre todo el vocabulario no escala con muchos requests.
- **Data drift:** sin retraining automatizado, la calidad decrece.
- **Seguridad:** exposición sin autenticación no es segura para producción.

---

## Siguiente Paso Propuesto

Dime cuál quieres que haga primero y lo implemento:

- **A)** Agrego endpoints `/health` y `/reload-models` a `api.py`, con tests.
- **B)** Mejoro `predict_ncf` para candidate sampling rápido y un modo “fast” que usa top-popular + NCF re-ranking.
- **C)** Creo GitHub Actions básico que corre tests y build de Docker.
- **D)** Escribo tests unitarios para `utils.py` y `predict.py`.
