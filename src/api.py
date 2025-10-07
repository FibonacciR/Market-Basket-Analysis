from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from src.predict import (
    load_ncf_model, predict_ncf,
    load_lgb_model, predict_lgb,
    load_association_rules, predict_association
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
    return {"message": "API de recomendaciones activa. Usa /predict/ncf, /predict/lgb o /predict/association para obtener recomendaciones."}

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

# Endpoint para Association Rules
@app.post("/predict/association")
def api_predict_association(request: PredictRequest):
    rules = load_association_rules()
    if rules is None:
        return {"error": "Association Rules model not found."}
    recs = predict_association(rules, request.user_id, request.basket)
    return {
        "model": "AssociationRules",
        "user_id": request.user_id,
        "basket": request.basket,
        "recommendation": recs
    }