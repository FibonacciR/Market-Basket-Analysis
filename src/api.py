from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from src.predict import (
    load_ncf_model, predict_ncf,
    load_lgb_model, predict_lgb
)
from src.utils import load_product_names

app = FastAPI(title="Market Basket Recommendation API",
              description="API para generar recomendaciones de productos usando modelos ML y DL",
              version="1.0.0")

class PredictRequest(BaseModel):
    user_id: int
    basket: List[int]

@app.get("/")
def root():
    return {"message": "API de recomendaciones activa. Usa /predict/ncf o /predict/lgb para obtener recomendaciones."}

# Endpoint para NCF
@app.post("/predict/ncf")
def api_predict_ncf(request: PredictRequest):
    model = load_ncf_model()
    if model is None:
        return {"error": "NCF model not found."}
    if isinstance(model, dict):
        return {
            "model": "NCF",
            "user_id": request.user_id,
            "basket": request.basket,
            "recommendation": {"error": "El modelo NCF cargado es un dict y no puede usarse para predicción. Verifica el archivo del modelo."}
        }

    res = predict_ncf(model, request.user_id, request.basket, top_k=10)
    if isinstance(res, dict) and 'error' in res:
        return {"model": "NCF", "user_id": request.user_id, "basket": request.basket, "recommendation": res}

    input_scores = res.get('input_scores', [])
    recommended = res.get('recommended', [])

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