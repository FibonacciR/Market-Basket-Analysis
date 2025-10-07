import os
import torch
import pickle
from typing import Optional
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ============================================================
#  NCF MODEL
# ============================================================

def load_ncf_model():
    path = os.path.join(MODELS_DIR, 'best_ncf_model.pt')
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    from src.model import NCF

    num_users = 131209
    num_products = 47913

    model = NCF(num_users=num_users, num_products=num_products)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_ncf(model, user_id, basket, top_k: int = 10, num_products_default: int = 47913):
    """
    Score candidate products with the NCF model and return recommendations.

    Parameters:
        model: torch model (NCF)
        user_id: int
        basket: list[int] (product ids to exclude from recommendations)
        top_k: number of recommendations to return

    Returns:
        dict with keys:
        - input_scores: list of scores for the provided basket (aligned)
        - recommended: list of {product_id, score, probability, product_name}
        Or returns {'error': '...'} on failure.
    """
    try:
        if isinstance(model, dict):
            return {"error": "El modelo NCF cargado es un dict y no puede usarse para predicción. Verifica el archivo del modelo."}

        # Use model's actual product vocabulary (0 to num_products-1)
        candidate_ids = list(range(min(num_products_default, model.product_embedding.num_embeddings)))

        # Load product names for display
        try:
            from src.utils import load_product_names
            product_names = load_product_names()
        except Exception:
            product_names = {}

        # Compute scores for the input basket
        input_scores = []
        if basket:
            user_tensor = torch.tensor([user_id] * len(basket), dtype=torch.long)
            basket_tensor = torch.tensor(basket, dtype=torch.long)
            with torch.no_grad():
                out = model(user_tensor, basket_tensor)
                input_scores = [float(x) for x in out.tolist()]

        # Score all candidate products
        device = torch.device('cpu')
        model.to(device)
        model.eval()

        batch_size = 4096
        scores = []

        with torch.no_grad():
            for i in range(0, len(candidate_ids), batch_size):
                chunk = candidate_ids[i:i + batch_size]
                u = torch.tensor([user_id] * len(chunk), dtype=torch.long, device=device)
                p = torch.tensor(chunk, dtype=torch.long, device=device)
                out = model(u, p)
                scores.extend([float(x) for x in out.tolist()])

        # Convert raw scores to probabilities via softmax
        scores_arr = np.array(scores, dtype=float)
        if scores_arr.size:
            max_s = np.max(scores_arr)
            exp_s = np.exp(scores_arr - max_s)
            probs = exp_s / np.sum(exp_s)
        else:
            probs = np.array([])

        # Build recommendation list excluding items in basket
        basket_set = set(int(x) for x in (basket or []))
        recs = []
        for pid, sc, pr in zip(candidate_ids, scores_arr.tolist(), probs.tolist() if probs.size else [0.0] * len(candidate_ids)):
            if pid in basket_set:
                continue
            recs.append({
                'product_id': int(pid),
                'score': float(sc),
                'probability': float(pr),
                'product_name': product_names.get(pid, '')
            })

        # Sort by probability desc and take top_k
        recs_sorted = sorted(recs, key=lambda x: x['probability'], reverse=True)[:top_k]

        return {'input_scores': input_scores, 'recommended': recs_sorted}

    except Exception as e:
        return {"error": f"NCF prediction error: {str(e)}"}


# ============================================================
#  LIGHTGBM MODEL
# ============================================================

def load_lgb_model():
    path = os.path.join(MODELS_DIR, 'baseline_lgb_model.txt')
    if not os.path.exists(path):
        return None
    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=path)
    except Exception:
        return None


def predict_lgb(model, user_id, basket):
    """
    Example placeholder for LightGBM prediction.
    """
    if model is None:
        return {"error": "LightGBM model not loaded."}
    try:
        # Construct a dummy feature vector
        features = np.array([[user_id] + basket])
        preds = model.predict(features)
        top_indices = np.argsort(preds)[::-1][:3]
        return top_indices.tolist()
    except Exception as e:
        return {"error": f"LightGBM prediction error: {str(e)}"}


# ============================================================
#  ASSOCIATION RULES
# ============================================================

def load_association_rules():
    path = os.path.join(MODELS_DIR, 'association_rules.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_association(rules, user_id, basket):
    """
    Predict recommendations using association rules.
    """
    try:
        if hasattr(rules, 'recommend') and callable(rules.recommend):
            return rules.recommend(basket)

        recommendations = []
        for rule in rules:
            if isinstance(rule, dict) and 'antecedents' in rule and 'consequents' in rule:
                if set(rule['antecedents']).issubset(set(basket)):
                    recommendations.extend(rule['consequents'])
        return list(set(recommendations))[:3]
    except Exception as e:
        return {"error": f"Association prediction error: {str(e)}"}
