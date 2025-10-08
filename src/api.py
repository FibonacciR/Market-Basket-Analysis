import os

# Crear el folder 'storage' y el archivo de credenciales si existe la variable de entorno
os.makedirs("storage", exist_ok=True)
gcs_key_env = os.getenv("GCS_KEY_JSON")
gcs_key_path = "storage/gcs-key.json"
print(f"DEBUG: GCS_KEY_JSON exists: {bool(gcs_key_env)}")
print(f"DEBUG: File exists: {os.path.exists(gcs_key_path)}")
if gcs_key_env and not os.path.exists(gcs_key_path):
    with open(gcs_key_path, "w") as f:
        f.write(gcs_key_env)
    print(f"DEBUG: File created successfully: {os.path.exists(gcs_key_path)}")
else:
    print("DEBUG: File not created - variable missing or file already exists")
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

@app.post("/download-models")
def download_models():
    """
    Download ML models and datasets from external Cloud Storage.
    Usage: POST /download-models - No parameters required.
    Downloads: NCF model, LightGBM model, products.csv to local directories.
    """
    try:
        GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "models_dl")
        GCS_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "storage/gcs-key.json")
        
        # Lista de archivos para descargar (modelos + datos)
        FILES_TO_DOWNLOAD = [
            {"source": "mba-api/baseline_lgb_model.txt", "dest_dir": "models"},
            {"source": "mba-api/best_ncf_model.pt", "dest_dir": "models"},
            {"source": "datasets/products.csv", "dest_dir": "data"}
        ]
        
        # Crear directorios necesarios
        for item in FILES_TO_DOWNLOAD:
            dest_dir = os.getenv("MODELS_DIR", item["dest_dir"]) if item["dest_dir"] == "models" else item["dest_dir"]
            os.makedirs(dest_dir, exist_ok=True)
        
        storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS)
        bucket = storage_client.bucket(GCS_BUCKET)
        results = []
        
        for item in FILES_TO_DOWNLOAD:
            try:
                blob = bucket.blob(item["source"])
                dest_dir = os.getenv("MODELS_DIR", item["dest_dir"]) if item["dest_dir"] == "models" else item["dest_dir"]
                
                # Preserve the original directory structure for models
                if item["dest_dir"] == "models":
                    dest_path = os.path.join(dest_dir, item["source"])  # Keep mba-api/ structure
                else:
                    filename = os.path.basename(item["source"])
                    dest_path = os.path.join(dest_dir, filename)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                blob.download_to_filename(dest_path)
                results.append({"file": item["source"], "destination": dest_path, "status": "ok"})
            except Exception as e:
                results.append({"file": item["source"], "status": "error", "detail": str(e)})
        
        success_count = sum(1 for r in results if r["status"] == "ok")
        error_count = sum(1 for r in results if r["status"] == "error")
        message = f"Download finished: {success_count} files OK, {error_count} errors."
        if error_count == 0:
            message += " All files were downloaded successfully."
        else:
            message += " Some files could not be downloaded."
            
        # Auto-reload models cache after successful download
        if success_count > 0:
            try:
                from src.utils import load_product_names
                load_product_names.cache_clear()
                message += " Model caches cleared automatically."
                log_structured("INFO", "Models cache auto-cleared after download", files_downloaded=success_count)
            except Exception as cache_error:
                log_structured("WARNING", "Failed to auto-clear cache after download", error=str(cache_error))
                message += f" Warning: Cache clear failed - {str(cache_error)}"
        
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
    """
    API welcome endpoint - provides basic information about available services.
    Usage: GET / - No parameters required.
    """
    return {"message": "Market Basket API is active. Use /predict/ncf or /predict/lgb for recommendations."}

@app.get("/health")
def health():
    """
    Health check endpoint - returns API status and uptime.
    Usage: GET /health - No parameters required.
    Returns: status, timestamp, uptime in seconds.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - metrics["start_time"])
    }

@app.get("/metrics")
def get_metrics():
    """
    API metrics endpoint - returns usage statistics and performance data.
    Usage: GET /metrics - No parameters required.
    Returns: request counts, error counts, uptime.
    """
    return {
        "requests_total": metrics["requests_total"],
        "requests_ncf": metrics["requests_ncf"], 
        "requests_association": metrics["requests_association"],
        "errors": metrics["errors"],
        "uptime_seconds": int(time.time() - metrics["start_time"])
    }

@app.post("/reload-models")
def reload_models():
    """
    Clear model caches to force reload on next prediction request.
    Usage: POST /reload-models - No parameters required.
    Useful after downloading new models or updating configurations.
    """
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

@app.post("/predict/ncf")
def api_predict_ncf(request: PredictRequest):
    """
    Neural Collaborative Filtering predictions for market basket analysis.
    Usage: POST /predict/ncf with JSON body: {"user_id": int, "basket": [product_ids]}
    Returns: top 10 product recommendations with scores and product names.
    """
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

@app.post("/predict/lgb")
def api_predict_lgb(request: PredictRequest):
    """
    LightGBM gradient boosting predictions for market basket analysis.
    Usage: POST /predict/lgb with JSON body: {"user_id": int, "basket": [product_ids]}
    Returns: product recommendations with probability scores and product names.
    """
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