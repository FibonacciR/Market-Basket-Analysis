import os

# Crear el folder 'storage' y el archivo de credenciales si existe la variable de entorno
os.makedirs("storage", exist_ok=True)
gcs_key_env = os.getenv("GCS_KEY_JSON")
gcs_key_path = "storage/gcs-key.json"
if gcs_key_env and not os.path.exists(gcs_key_path):
    with open(gcs_key_path, "w") as f:
        f.write(gcs_key_env)
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
from google.cloud import storage
import time
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from src.predict import (
    load_ncf_model, predict_ncf,
    load_lgb_model, predict_lgb,
    load_association_rules, predict_association
)
from src.utils import load_product_names

app = FastAPI(title="Market Basket Recommendation API",
              description="API para generar recomendaciones de productos usando modelos ML y DL",
              version="1.0.0")
# ...existing endpoints...

# Endpoint para descargar modelos desde GCS
@app.post("/download-models")
def download_models():
    try:
        GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "models_dl")
        GCS_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "storage/gcs-key.json")
        MODELS = [
            "mba-api/baseline_lgb_model.txt",
            "mba-api/best_ncf_model.pt"
        ]
        MODELS_DIR = os.getenv("MODELS_DIR", "models")
        os.makedirs(MODELS_DIR, exist_ok=True)
        storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS)
        bucket = storage_client.bucket(GCS_BUCKET)
        results = []
        for model_name in MODELS:
            try:
                blob = bucket.blob(model_name)
                dest_path = os.path.join(MODELS_DIR, model_name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                blob.download_to_filename(dest_path)
                results.append({"model": model_name, "status": "ok"})
            except Exception as e:
                results.append({"model": model_name, "status": "error", "detail": str(e)})
        success_count = sum(1 for r in results if r["status"] == "ok")
        error_count = sum(1 for r in results if r["status"] == "error")
        message = f"Download finished: {success_count} models OK, {error_count} errors."
        if error_count == 0:
            message += " All models were downloaded successfully."
        else:
            message += " Some models could not be downloaded."
        return JSONResponse(content={"message": message, "results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Configurar logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def log_structured(level: str, message: str, **kwargs):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        **kwargs
    }
    logger.info(json.dumps(log_data))

# Métricas simples en memoria
metrics = {
    "requests_total": 0,
    "requests_ncf": 0,
    "requests_association": 0,
    "errors": 0,
    "start_time": time.time()
}

class PredictRequest(BaseModel):
    user_id: int
    basket: List[int]

@app.get("/")
def root():
    return {"message": "API de recomendaciones activa. Usa /predict/ncf o /predict/lgb para obtener recomendaciones."}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - metrics["start_time"])
    }

@app.get("/metrics")
def get_metrics():
    return {
        "requests_total": metrics["requests_total"],
        "requests_ncf": metrics["requests_ncf"], 
        "requests_association": metrics["requests_association"],
        "errors": metrics["errors"],
        "uptime_seconds": int(time.time() - metrics["start_time"])
    }

@app.post("/reload-models")
def reload_models():
    try:
        # Limpiar cache de load_product_names
        from src.utils import load_product_names
        load_product_names.cache_clear()
        
        # Forzar recarga en próxima llamada
        log_structured("INFO", "Models cache cleared successfully")
        
        return {
            "status": "success",
            "message": "Model caches cleared. Models will be reloaded on next request.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        log_structured("ERROR", "Failed to reload models", error=str(e))
        return {
            "status": "error", 
            "message": f"Failed to reload models: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Endpoint para NCF
@app.post("/predict/ncf")
def api_predict_ncf(request: PredictRequest):
    start_time = time.time()
    metrics["requests_total"] += 1
    metrics["requests_ncf"] += 1
    
    log_structured("INFO", "NCF prediction request", 
                  user_id=request.user_id, basket_size=len(request.basket))
    
    model = load_ncf_model()
    if model is None:
        metrics["errors"] += 1
        log_structured("ERROR", "NCF model not found", user_id=request.user_id)
        return {"error": "NCF model not found."}
    if isinstance(model, dict):
        metrics["errors"] += 1
        log_structured("ERROR", "NCF model is dict", user_id=request.user_id)
        return {
            "model": "NCF",
            "user_id": request.user_id,
            "basket": request.basket,
            "recommendation": {"error": "El modelo NCF cargado es un dict y no puede usarse para predicción. Verifica el archivo del modelo."}
        }

    res = predict_ncf(model, request.user_id, request.basket, top_k=10)
    if isinstance(res, dict) and 'error' in res:
        metrics["errors"] += 1
        log_structured("ERROR", "NCF prediction failed", user_id=request.user_id, error=res.get('error'))
        return {"model": "NCF", "user_id": request.user_id, "basket": request.basket, "recommendation": res}

    input_scores = res.get('input_scores', [])
    recommended = res.get('recommended', [])

    # Log successful response
    duration = time.time() - start_time
    log_structured("INFO", "NCF prediction completed", 
                  user_id=request.user_id, recommendations=len(recommended), 
                  duration_ms=int(duration * 1000))

    product_names = load_product_names()
    scores_aligned = []
    for pid, sc in zip(request.basket, input_scores):
        scores_aligned.append({
            'product_id': int(pid),
            'score': float(sc),
            'product_name': product_names.get(int(pid), ''),
        })

    recommended_ids = [r['product_id'] for r in recommended]

    return {
        "model": "NCF",
        "user_id": request.user_id,
        "basket": request.basket,
        "recommendation": {
            "scores": scores_aligned,
            "recommended": recommended,
            "recommended_ids": recommended_ids
        }
    }

# Endpoint para LightGBM
@app.post("/predict/lgb")
def api_predict_lgb(request: PredictRequest):
    model = load_lgb_model()
    if model is None:
        return {"error": "LightGBM model not found."}
    res = predict_lgb(model, request.user_id, request.basket)
    if isinstance(res, dict) and 'error' in res:
        return {"model": "LightGBM", "user_id": request.user_id, "basket": request.basket, "recommendation": res}

    # Usar la misma estructura que NCF
    scores_aligned = res.get('scores', [])
    recommended = res.get('recommended', [])
    recommended_ids = res.get('recommended_ids', [])

    return {
        "model": "LightGBM",
        "user_id": request.user_id,
        "basket": request.basket,
        "recommendation": {
            "scores": scores_aligned,
            "recommended": recommended,
            "recommended_ids": recommended_ids
        }
    }